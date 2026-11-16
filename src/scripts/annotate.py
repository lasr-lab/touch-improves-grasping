#!/usr/bin/env python3
"""Interactive success/failure/delete annotator for episode videos.

Walks a dataset root, treats every folder that directly contains a `camera/`
subfolder as one episode, plays its `camera/rgb/rgb.mp4`, and on a keypress
moves the whole episode folder into `<group>/success/`, `<group>/failure/` or
`<group>/delete/`, where <group> is the episode's parent folder (e.g.
`act/nontac`). All three are plain folder moves, so every decision — including
a delete — can be undone with U.

Every move is appended to `annotation_log.csv` in the root, so a session can be
audited or reverted later even after the GUI is closed.

Usage:
    python3 annotate.py                       # annotate everything under cwd
    python3 annotate.py --root act/nontac     # one group only
    python3 annotate.py --scan-only           # print what was found and exit
    python3 annotate.py --dry-run             # play + decide, move nothing
"""

import argparse
import csv
import os
import re
import shutil
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

SUCCESS = "success"
FAILURE = "failure"
DELETE = "delete"
LABELS = (SUCCESS, FAILURE, DELETE)

# Button/label presentation. Delete stays glyph-free because the trash emoji is
# not in the fonts Tk reliably finds on Linux.
LABEL_TITLES = {SUCCESS: "SUCCESS", FAILURE: "FAILURE", DELETE: "DELETE"}
LABEL_KEYS = {SUCCESS: "S / →", FAILURE: "F / ←", DELETE: "D"}
LABEL_COLORS = {
    SUCCESS: ("#1f7a3d", "#28994e"),
    FAILURE: ("#a32626", "#c53030"),
    DELETE: ("#3d3d3d", "#5a5a5a"),
}

EPISODE_MARKER = "camera"
VIDEO_NAME = "rgb.mp4"
LOG_NAME = "annotation_log.csv"
LOG_FIELDS = ["timestamp", "action", "label", "group", "episode", "src", "dst"]

# Never descend into these: they hold the per-episode frame dumps (hundreds of
# JPEGs each on a slow drive) or are annotation output we must not re-discover.
SKIP_DIR_NAMES = {
    EPISODE_MARKER,
    "action",
    "state",
    "digit360",
    "frames",
    SUCCESS,
    FAILURE,
    DELETE,
    "__pycache__",
}

SPEEDS = (0.5, 1.0, 2.0, 3.0, 4.0)
IDLE_POLL_MS = 100

_NUM_RE = re.compile(r"(\d+)")


def natural_key(text):
    """Sort key that orders 0, 1, 2, 10 instead of 0, 1, 10, 2."""
    parts = [p for p in _NUM_RE.split(str(text)) if p != ""]
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p.lower()) for p in parts)


@dataclass
class Episode:
    path: Path
    video: Path
    group: Path


@dataclass
class GroupState:
    path: Path
    episodes: list
    prior: dict


def find_video(episode_dir):
    """Locate the playback video for an episode, or None if there is none."""
    camera = episode_dir / EPISODE_MARKER
    preferred = camera / "rgb" / VIDEO_NAME
    if preferred.is_file():
        return preferred
    if not camera.is_dir():
        return None
    # Fallback for episodes encoded to a different name: only look one or two
    # levels down, never walk the whole frames tree.
    for search_dir in (camera / "rgb", camera):
        try:
            entries = list(os.scandir(search_dir))
        except OSError:
            continue
        videos = [e for e in entries if e.is_file() and e.name.lower().endswith(".mp4")]
        if videos:
            videos.sort(key=lambda e: natural_key(e.name))
            return Path(videos[0].path)
    return None


def count_annotated(group_dir):
    """Count episodes already filed under each label folder for a group."""
    prior = {label: 0 for label in LABELS}
    for label in LABELS:
        target = group_dir / label
        try:
            with os.scandir(target) as it:
                prior[label] = sum(1 for e in it if e.is_dir())
        except OSError:
            pass
    return prior


def scan_root(root, limit=None):
    """Find every episode under root without descending into frame dumps.

    Returns (groups, missing) where groups is a list of GroupState ordered by
    relative path, and missing lists episode folders with no playable video.

    The drive this runs on is slow and each episode holds hundreds of loose
    JPEGs, so the hot path probes `<child>/camera/rgb/rgb.mp4` with a single
    stat and only falls back to listing a directory when that probe misses.
    """
    root = Path(root)
    found = {}
    missing = []
    stack = [root]

    def record(episode_dir, video):
        if video is None:
            missing.append(episode_dir)
        else:
            found.setdefault(episode_dir.parent, []).append(
                Episode(episode_dir, video, episode_dir.parent)
            )

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                subdirs = [e for e in it if e.is_dir(follow_symlinks=False)]
        except OSError:
            continue

        if any(e.name == EPISODE_MARKER for e in subdirs):
            # `current` is an episode folder: record it and do not descend.
            record(current, find_video(current))
            continue

        for entry in subdirs:
            if entry.name in SKIP_DIR_NAMES or entry.name.startswith("."):
                continue
            child = Path(entry.path)
            expected = child / EPISODE_MARKER / "rgb" / VIDEO_NAME
            if expected.is_file():
                record(child, expected)
            else:
                stack.append(child)

    groups = []
    for group_dir in sorted(found, key=lambda p: tuple(natural_key(x) for x in p.relative_to(root).parts)):
        episodes = sorted(found[group_dir], key=lambda e: natural_key(e.path.name))
        if limit is not None:
            episodes = episodes[:limit]
        groups.append(GroupState(group_dir, episodes, count_annotated(group_dir)))
    missing.sort(key=lambda p: tuple(natural_key(x) for x in p.relative_to(root).parts))
    return groups, missing


def rel(root, path):
    try:
        result = str(Path(path).relative_to(root))
    except ValueError:
        return str(path)
    # The root itself would otherwise display as "." in the group picker.
    return Path(path).name if result == "." else result


class VideoPlayer:
    """Streams an mp4 through OpenCV onto a Tk canvas item."""

    def __init__(self, canvas, image_item, on_end, on_progress):
        self.canvas = canvas
        self.image_item = image_item
        self.on_end = on_end
        self.on_progress = on_progress
        self.cap = None
        self.path = None
        self.fps = 30.0
        self.frame_count = 0
        self.frame_index = 0
        self.speed_index = SPEEDS.index(1.0)
        self.playing = False
        self.raw_frame = None
        self.video_size = (1, 1)
        self.display_size = (1, 1)
        self.photo = None
        self._after_id = None
        self._next_frame_at = 0.0
        self._frame_interval = 1.0 / 30.0
        self._closed = False

    @property
    def speed(self):
        return SPEEDS[self.speed_index]

    def open(self, path):
        self.close()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.cap = None
            return False
        self.cap = cap
        self.path = Path(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_index = -1
        self.raw_frame = None
        ok, frame = cap.read()
        if ok:
            self.frame_index = 0
            self.raw_frame = frame
            self.video_size = (frame.shape[1], frame.shape[0])
        return True

    def close(self):
        self.playing = False
        if self._after_id is not None and self.canvas is not None:
            try:
                self.canvas.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def shutdown(self):
        self._closed = True
        self.close()

    def fit_to(self, width, height):
        """Recompute the display size for the given canvas box and redraw."""
        if width < 16 or height < 16:
            return
        vw, vh = self.video_size
        scale = min(width / vw, height / vh)
        size = (max(1, int(vw * scale)), max(1, int(vh * scale)))
        if size != self.display_size:
            self.display_size = size
            self.render()

    def render(self):
        if self.raw_frame is None or self.canvas is None:
            return
        rgb = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if self.display_size != image.size:
            image = image.resize(self.display_size, Image.BILINEAR)
        # Keep a reference: Tk drops the image as soon as this goes out of scope.
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def play(self):
        if self.cap is None:
            return
        if self.frame_index >= self.frame_count - 1:
            self.restart()
        self.playing = True
        self._next_frame_at = time.perf_counter()
        self._schedule()

    def pause(self):
        self.playing = False
        self._schedule()

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def restart(self):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_index = -1
        self._read_frame()
        self.play()

    def seek_frames(self, index):
        if self.cap is None:
            return
        index = max(0, min(index, max(0, self.frame_count - 1)))
        was_playing = self.playing
        self.playing = False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.frame_index = index - 1
        self._read_frame()
        self.on_progress()
        if was_playing:
            self.play()
        else:
            self._schedule()

    def seek_seconds(self, delta):
        self.seek_frames(self.frame_index + int(delta * self.fps))

    def nudge_speed(self, delta):
        self.speed_index = max(0, min(len(SPEEDS) - 1, self.speed_index + delta))
        if self.playing:
            self._schedule()

    def _read_frame(self):
        """Advance the capture by `speed` video frames and store the last one."""
        steps = max(1, int(round(self.speed)))
        for _ in range(steps - 1):
            if not self.cap.grab():
                return False
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.frame_index += steps
        self.raw_frame = frame
        return True

    def _schedule(self):
        if self._closed or self.canvas is None:
            return
        if self._after_id is not None:
            try:
                self.canvas.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.playing:
            steps = max(1, int(round(self.speed)))
            interval = steps / (self.fps * self.speed)
            delay = max(1, int((self._next_frame_at - time.perf_counter()) * 1000))
            self._frame_interval = interval
        else:
            delay = IDLE_POLL_MS
        self._after_id = self.canvas.after(delay, self._tick)

    def _tick(self):
        if self._closed:
            return
        self._after_id = None
        if not self.playing or self.cap is None:
            self._schedule()
            return
        if not self._read_frame():
            self.playing = False
            self.on_progress()
            self.on_end()
            self._schedule()
            return
        self.render()
        self.on_progress()
        # Advance on the scheduled grid so per-tick Tk overhead is absorbed by
        # the next delay instead of being added to it (which would play slow).
        # Only resync after a real stall, and never burst to catch up.
        self._next_frame_at += self._frame_interval
        now = time.perf_counter()
        if self._next_frame_at < now:
            self._next_frame_at = now + self._frame_interval
        self._schedule()


class AnnotatorApp:
    def __init__(self, root, groups, missing, dataset_root, args):
        self.root = root
        self.args = args
        self.dataset_root = Path(dataset_root)
        self.missing = missing
        self.groups = groups
        self.group_index = 0
        self.cursor = 0
        self.session_counts = {label: 0 for label in LABELS}
        self.undo_stack = []
        self.log_path = self.dataset_root / LOG_NAME
        self._loading = False

        root.title(f"Episode annotator — {self.dataset_root}")
        root.geometry(f"{args.width + 120}x{args.height + 250}")
        root.minsize(560, 420)

        self._build_ui()
        self._bind_keys()

        self.player = VideoPlayer(
            self.canvas, self.canvas_image, self._on_video_end, self._update_seek
        )

        self._write_log_header()
        self._refresh_group_menu()
        if missing:
            names = ", ".join(rel(self.dataset_root, p) for p in missing[:3])
            extra = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            self.status_var.set(
                f"{len(missing)} episode(s) have no playable .mp4 and were skipped: {names}{extra}"
            )
        self.root.after(50, self._load_current)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI construction ----------

    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        top = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Group:").pack(side=tk.LEFT)
        self.group_var = tk.StringVar()
        self.group_menu = ttk.Combobox(
            top, textvariable=self.group_var, state="readonly", width=34, takefocus=0
        )
        self.group_menu.pack(side=tk.LEFT, padx=(6, 14))
        self.group_menu.bind("<<ComboboxSelected>>", self._on_group_selected)

        self.position_var = tk.StringVar(value="—")
        ttk.Label(top, textvariable=self.position_var, font=("", 10, "bold")).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(top, mode="determinate", length=180)
        self.progress.pack(side=tk.RIGHT)

        self.path_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.path_var, anchor="w", foreground="#555").pack(
            side=tk.TOP, fill=tk.X, padx=12
        )

        self.canvas = tk.Canvas(self.root, background="#101010", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", lambda e: self.player.toggle())

        self.hint = tk.Label(
            self.canvas,
            text="",
            background="#101010",
            foreground="#dddddd",
            font=("", 11, "bold"),
        )

        transport = ttk.Frame(self.root, padding=(10, 0, 10, 4))
        transport.pack(side=tk.TOP, fill=tk.X)

        for text, command in (
            ("⟲ Replay (R)", self.player_restart),
            ("⏯ Play/Pause (Space)", self.player_toggle),
            ("−1s (,)", lambda: self.player.seek_seconds(-1)),
            ("+1s (.)", lambda: self.player.seek_seconds(1)),
        ):
            ttk.Button(transport, text=text, command=command, takefocus=0).pack(
                side=tk.LEFT, padx=(0, 6)
            )

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek = ttk.Scale(
            transport, from_=0.0, to=1.0, variable=self.seek_var, takefocus=0, length=260
        )
        self.seek.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.seek.bind("<ButtonRelease-1>", self._on_seek_release)

        self.time_var = tk.StringVar(value="0.0 / 0.0s")
        ttk.Label(transport, textvariable=self.time_var, width=16, anchor="e").pack(side=tk.LEFT)

        self.speed_var = tk.StringVar(value="1.0x")
        ttk.Button(
            transport, textvariable=self.speed_var, command=self.cycle_speed, width=6, takefocus=0
        ).pack(side=tk.LEFT, padx=(6, 0))

        buttons = ttk.Frame(self.root, padding=(10, 4, 10, 6))
        buttons.pack(side=tk.TOP, fill=tk.X)

        self.label_buttons = {}
        for index, label in enumerate(LABELS):
            background, active = LABEL_COLORS[label]
            button = tk.Button(
                buttons,
                text=f"{LABEL_TITLES[label]}   ({LABEL_KEYS[label]})",
                command=lambda chosen=label: self.label(chosen),
                background=background,
                foreground="white",
                activebackground=active,
                activeforeground="white",
                font=("", 13, "bold"),
                relief=tk.FLAT,
                pady=10,
                takefocus=0,
            )
            padx = (0, 6) if index == 0 else ((6, 6) if index < len(LABELS) - 1 else (6, 0))
            button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=padx)
            self.label_buttons[label] = button

        aux = ttk.Frame(self.root, padding=(10, 0, 10, 4))
        aux.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(aux, text="Skip (N)", command=self.skip, takefocus=0).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(aux, text="◀ Prev (P)", command=self.previous, takefocus=0).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(aux, text="Undo (U)", command=self.undo, takefocus=0).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(aux, text="Quit (Q)", command=self._on_close, takefocus=0).pack(side=tk.LEFT)
        ttk.Label(
            aux,
            text="speed: −/=    •    every move is logged to annotation_log.csv",
            foreground="#666",
        ).pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            self.root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN, padding=(8, 4)
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self):
        bindings = {
            "<s>": lambda: self.label(SUCCESS),
            "<S>": lambda: self.label(SUCCESS),
            "<f>": lambda: self.label(FAILURE),
            "<F>": lambda: self.label(FAILURE),
            "<d>": lambda: self.label(DELETE),
            "<D>": lambda: self.label(DELETE),
            "<Right>": lambda: self.label(SUCCESS),
            "<Left>": lambda: self.label(FAILURE),
            "<r>": self.player_restart,
            "<R>": self.player_restart,
            "<space>": self.player_toggle,
            "<n>": self.skip,
            "<N>": self.skip,
            "<p>": self.previous,
            "<P>": self.previous,
            "<u>": self.undo,
            "<U>": self.undo,
            "<BackSpace>": self.undo,
            "<comma>": lambda: self.player.seek_seconds(-1),
            "<period>": lambda: self.player.seek_seconds(1),
            "<minus>": lambda: self.change_speed(-1),
            "<underscore>": lambda: self.change_speed(-1),
            "<equal>": lambda: self.change_speed(1),
            "<plus>": lambda: self.change_speed(1),
            "<q>": self._on_close,
            "<Q>": self._on_close,
            "<Escape>": self._on_close,
        }
        for sequence, handler in bindings.items():
            self.root.bind_all(sequence, self._guard(handler), add="+")

    @staticmethod
    def _guard(handler):
        """Run a binding and swallow its event so focused widgets don't also act."""

        def wrapper(event):
            handler()
            return "break"

        return wrapper

    # ---------- state helpers ----------

    def _current_group(self):
        if not self.groups or self.group_index >= len(self.groups):
            return None
        return self.groups[self.group_index]

    def _current_episode(self):
        group = self._current_group()
        if group is None or not group.episodes:
            return None
        if self.cursor < 0 or self.cursor >= len(group.episodes):
            return None
        return group.episodes[self.cursor]

    def _tally(self, counts):
        """Compact one-line summary of a label->count mapping."""
        return f"✓{counts[SUCCESS]} ✗{counts[FAILURE]} ⌫{counts[DELETE]}"

    def _session_totals(self):
        """Episodes already filed, counting prior runs plus this session."""
        totals = {label: 0 for label in LABELS}
        for group in self.groups:
            for label in LABELS:
                totals[label] += group.prior[label]
        return totals

    def _refresh_group_menu(self):
        labels = []
        for group in self.groups:
            remaining = len(group.episodes)
            labels.append(f"{rel(self.dataset_root, group.path)}  ({remaining} left)")
        self.group_menu["values"] = labels
        if self.groups:
            self.group_menu.current(min(self.group_index, len(self.groups) - 1))

    def _refresh_header(self):
        group = self._current_group()
        episode = self._current_episode()
        if group is None or episode is None:
            self.position_var.set("no episodes left")
            self.path_var.set("")
            self.progress.configure(value=0, maximum=1)
            return
        total = len(group.episodes)
        self.position_var.set(
            f"Episode {self.cursor + 1} / {total} remaining   "
            f"(session {self._tally(self.session_counts)})"
        )
        self.path_var.set(rel(self.dataset_root, episode.path))
        self.progress.configure(maximum=max(1, total), value=max(0, total - self.cursor))

    # ---------- playback ----------

    def _load_current(self):
        if self._loading:
            return
        self._loading = True
        try:
            episode = self._current_episode()
            self.hint.place_forget()
            if episode is None:
                self.canvas.itemconfigure(self.canvas_image, image="")
                self.player.close()
                self._refresh_header()
                self._advance_group_or_finish()
                return
            if not self.player.open(episode.video):
                self.status_var.set(f"Could not open {episode.video}")
                self._show_hint("⚠ video would not open — press N to skip")
                self._refresh_header()
                return
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            if width > 16 and height > 16:
                self.player.fit_to(width, height)
            else:
                self.player.fit_to(self.args.width, self.args.height)
            self.player.render()
            self._refresh_header()
            self._update_seek()
            if self.args.autoplay:
                self.player.play()
            else:
                self.player.pause()
        finally:
            self._loading = False

    def _show_hint(self, text):
        self.hint.configure(text=text)
        self.hint.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def _on_video_end(self):
        self._show_hint("ENDED — S success · F failure · D delete")
        self._update_seek()

    def _update_seek(self):
        total = max(1, self.player.frame_count)
        position = max(0, min(self.player.frame_index, self.player.frame_count))
        self.seek.configure(to=float(total))
        self.seek_var.set(float(position))
        self.time_var.set(f"{position / self.player.fps:.1f} / {self.player.frame_count / self.player.fps:.1f}s")

    def _on_seek_release(self, _event=None):
        self.hint.place_forget()
        self.player.seek_frames(int(self.seek_var.get()))

    def _on_canvas_resize(self, event):
        self.player.fit_to(event.width, event.height)
        if self.player.raw_frame is None:
            return
        self.canvas.itemcoords(
            self.canvas_image,
            (event.width - self.player.display_size[0]) // 2,
            (event.height - self.player.display_size[1]) // 2,
        )

    def player_restart(self):
        self.hint.place_forget()
        self.player.restart()

    def player_toggle(self):
        self.hint.place_forget()
        self.player.toggle()

    def change_speed(self, delta):
        self.player.nudge_speed(delta)
        self.speed_var.set(f"{self.player.speed:g}x")

    def cycle_speed(self):
        self.change_speed(1 if self.player.speed_index < len(SPEEDS) - 1 else -len(SPEEDS) + 1)

    # ---------- annotation ----------

    def _write_log_header(self):
        if self.log_path.exists():
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", newline="") as handle:
                csv.writer(handle).writerow(LOG_FIELDS)
        except OSError as exc:
            self.status_var.set(f"Warning: cannot write log ({exc})")

    def _log(self, action, label, src, dst):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "label": label,
            "group": rel(self.dataset_root, Path(src).parent if action == "move" else Path(dst).parent),
            "episode": Path(src).name if action == "move" else Path(dst).name,
            "src": str(src),
            "dst": str(dst),
        }
        try:
            with self.log_path.open("a", newline="") as handle:
                csv.writer(handle).writerow([row[field] for field in LOG_FIELDS])
        except OSError as exc:
            self.status_var.set(f"Warning: log write failed ({exc})")

    def label(self, which):
        episode = self._current_episode()
        if episode is None:
            self.status_var.set("Nothing to annotate in this group.")
            return
        if self.args.dry_run:
            self.session_counts[which] += 1
            self.status_var.set(f"[dry-run] would move {rel(self.dataset_root, episode.path)} → {which}/")
            group = self._current_group()
            del group.episodes[self.cursor]
            self.cursor = min(self.cursor, max(0, len(group.episodes) - 1))
            self._refresh_group_menu()
            self._load_current()
            return

        dest_dir = episode.group / which
        dest = dest_dir / episode.path.name
        if dest.exists():
            self.status_var.set(f"Skipped: {rel(self.dataset_root, dest)} already exists")
            self.skip()
            return

        self.player.pause()
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(episode.path), str(dest))
        except OSError as exc:
            messagebox.showerror("Move failed", f"{episode.path}\n→ {dest}\n\n{exc}")
            self.status_var.set("Move failed; episode left in place.")
            return

        self._log("move", which, episode.path, dest)
        self.undo_stack.append((which, episode.path, dest))
        self.session_counts[which] += 1

        group = self._current_group()
        group.prior[which] += 1
        del group.episodes[self.cursor]
        self.cursor = min(self.cursor, max(0, len(group.episodes) - 1))
        self.status_var.set(
            f"Moved {rel(self.dataset_root, episode.path)} → {which}/   "
            f"({self._tally(self.session_counts)} this session)"
        )
        self._refresh_group_menu()
        self._load_current()

    def undo(self):
        if not self.undo_stack:
            self.status_var.set("Nothing to undo.")
            return
        which, src, dest = self.undo_stack.pop()
        if self.args.dry_run:
            self.session_counts[which] = max(0, self.session_counts[which] - 1)
            self.status_var.set("[dry-run] undo is a no-op")
            return
        if not dest.exists():
            self.status_var.set(f"Undo failed: {dest} is gone (moved elsewhere?)")
            return
        try:
            shutil.move(str(dest), str(src))
        except OSError as exc:
            messagebox.showerror("Undo failed", f"{dest}\n→ {src}\n\n{exc}")
            self.undo_stack.append((which, src, dest))
            return
        self._log("undo", which, dest, src)
        self.session_counts[which] = max(0, self.session_counts[which] - 1)
        group = self._current_group()
        restored = Episode(src, find_video(src) or src, src.parent)
        group.episodes.insert(min(self.cursor, len(group.episodes)), restored)
        group.prior[which] = max(0, group.prior[which] - 1)
        self.status_var.set(f"Undo: {rel(self.dataset_root, src)} restored")
        self._refresh_group_menu()
        self._load_current()

    def skip(self):
        group = self._current_group()
        if group is None or not group.episodes:
            return
        episode = self._current_episode()
        if episode is None:
            return
        self.cursor = (self.cursor + 1) % len(group.episodes)
        self.status_var.set(f"Skipped {rel(self.dataset_root, episode.path)} (still pending, use P to go back)")
        self._load_current()

    def previous(self):
        group = self._current_group()
        if group is None or not group.episodes:
            return
        self.cursor = (self.cursor - 1) % len(group.episodes)
        self._load_current()

    def _advance_group_or_finish(self):
        remaining = [i for i, g in enumerate(self.groups) if g.episodes]
        if not remaining:
            self.player.close()
            self.status_var.set(
                f"All done. This session: {self._tally(self.session_counts)}."
            )
            self._show_hint("ALL EPISODES ANNOTATED")
            return
        if self.group_index in remaining:
            # Same group still has episodes; the cursor was just out of range.
            self.cursor = 0
            self._load_current()
            return
        # This group is exhausted: prefer the next one, else wrap to the first.
        later = [i for i in remaining if i > self.group_index]
        self.group_index = later[0] if later else remaining[0]
        self.cursor = 0
        self.group_menu.current(self.group_index)
        self.status_var.set(f"Group complete — moved on to {rel(self.dataset_root, self._current_group().path)}")
        self._load_current()

    def _on_group_selected(self, _event=None):
        index = self.group_menu.current()
        if index < 0 or index == self.group_index:
            return
        self.group_index = index
        self.cursor = 0
        self._load_current()

    def _on_close(self):
        self.player.shutdown()
        total = sum(len(g.episodes) for g in self.groups)
        message = (
            f"Filed this session: {self._tally(self.session_counts)}\n"
            f"Filed overall:      {self._tally(self._session_totals())}\n"
            f"Still unannotated:  {total}\n\n"
            f"Log: {self.log_path}"
        )
        try:
            if messagebox.askokcancel("Quit", message + "\n\nQuit?"):
                self.root.destroy()
        except tk.TclError:
            self.root.destroy()


def run_scan_only(groups, missing, dataset_root):
    total = sum(len(g.episodes) for g in groups)
    print(f"Root: {dataset_root}")
    print(f"Groups: {len(groups)}   Episodes with video: {total}   Episodes without video: {len(missing)}\n")
    header = f"{'group':<34}{'todo':>6}{'succ':>7}{'fail':>7}{'del':>6}"
    print(header)
    print("-" * len(header))
    for group in groups:
        counts = "".join(f"{group.prior[label]:>7}" for label in (SUCCESS, FAILURE))
        print(f"{rel(dataset_root, group.path):<34}{len(group.episodes):>6}{counts}{group.prior[DELETE]:>6}")
    print("-" * len(header))
    totals = "".join(
        f"{sum(g.prior[label] for g in groups):>7}" for label in (SUCCESS, FAILURE)
    )
    print(f"{'TOTAL':<34}{total:>6}{totals}{sum(g.prior[DELETE] for g in groups):>6}")
    if missing:
        print(f"\nEpisodes with no playable .mp4 ({len(missing)}):")
        for path in missing[:20]:
            print(f"  {rel(dataset_root, path)}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Watch each episode's rgb.mp4 and file the episode folder into success/, failure/ or delete/."
    )
    parser.add_argument("--root", default=".", help="Dataset root to scan (default: current directory)")
    parser.add_argument("--group", default=None, help="Only annotate the group whose path ends with this, e.g. act/nontac")
    parser.add_argument("--scan-only", action="store_true", help="Print the group/episode table and exit")
    parser.add_argument("--dry-run", action="store_true", help="Play and decide but move nothing")
    parser.add_argument("--limit", type=int, default=None, help="Cap episodes per group (for a trial run)")
    parser.add_argument("--no-autoplay", dest="autoplay", action="store_false", help="Do not auto-start each video")
    parser.add_argument("--width", type=int, default=1120, help="Initial video width in pixels")
    parser.add_argument("--height", type=int, default=630, help="Initial video height in pixels")
    args = parser.parse_args()

    dataset_root = Path(args.root).resolve()
    if not dataset_root.is_dir():
        sys.exit(f"Error: '{dataset_root}' is not a directory.")

    print(f"Scanning {dataset_root} ...")
    started = time.perf_counter()
    groups, missing = scan_root(dataset_root, args.limit)
    print(f"Scan took {time.perf_counter() - started:.1f}s")

    if args.group:
        needle = args.group.strip("/").replace("\\", "/")
        groups = [g for g in groups if rel(dataset_root, g.path).endswith(needle)]
        if not groups:
            sys.exit(f"Error: no group under '{dataset_root}' ends with '{needle}'.")

    if args.scan_only:
        run_scan_only(groups, missing, dataset_root)
        return

    if not groups or not any(g.episodes for g in groups):
        run_scan_only(groups, missing, dataset_root)
        print("\nNothing to annotate.")
        return

    root = tk.Tk()
    AnnotatorApp(root, groups, missing, dataset_root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
