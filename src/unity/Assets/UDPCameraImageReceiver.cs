using System;
using System.IO;
using System.IO.Compression;
using UnityEngine;
using UnityEngine.UI;

public class UDPCameraImageReceiver : MonoBehaviour
{
    [Header("Displays")]
    public RawImage colorImageDisplay;
    public RawImage depthImageDisplay;

    [Header("FPS UI (Optional)")]
    public Text colorFpsText;
    public Text depthFpsText;

    private Texture2D colorTexture;
    private Texture2D depthDisplayTexture;

    private ushort[] _depthData;
    private int _depthWidth;
    private int _depthHeight;
    private bool _depthUpdated;
    private readonly object _depthLock = new object();

    // FPS Tracking variables
    private int _colorFrameCount = 0;
    private int _depthFrameCount = 0;
    private float _fpsTimer = 0f;

    public int CurrentColorFPS { get; private set; }
    public int CurrentDepthFPS { get; private set; }

    public bool HasNewDepthData
    {
        get { lock (_depthLock) { return _depthUpdated; } }
    }

    void Start()
    {
        colorTexture = new Texture2D(2, 2);
        depthDisplayTexture = new Texture2D(2, 2);

        UDPManager.Instance.OnColorImageReceived += HandleColorImage;
        UDPManager.Instance.OnDepthImageReceived += HandleDepthImage;
    }

    void OnDestroy()
    {
        if (UDPManager.Instance != null)
        {
            UDPManager.Instance.OnColorImageReceived -= HandleColorImage;
            UDPManager.Instance.OnDepthImageReceived -= HandleDepthImage;
        }
    }

    void Update()
    {
        _fpsTimer += Time.deltaTime;
        if (_fpsTimer >= 1.0f)
        {
            CurrentColorFPS = _colorFrameCount;
            CurrentDepthFPS = _depthFrameCount;

            if (colorFpsText != null)
                colorFpsText.text = $"Color: {CurrentColorFPS} FPS";

            if (depthFpsText != null)
                depthFpsText.text = $"Depth: {CurrentDepthFPS} FPS";

            _colorFrameCount = 0;
            _depthFrameCount = 0;
            _fpsTimer -= 1.0f;
        }
    }

    private void HandleColorImage(ImageData data)
    {
        byte[] imageBytes = Convert.FromBase64String(data.data);
        colorTexture.LoadImage(imageBytes);

        if (colorImageDisplay != null)
        {
            colorImageDisplay.texture = colorTexture;
        }

        _colorFrameCount++;
    }

    private async void HandleDepthImage(DepthImageData data)
    {
        byte[] compressed = Convert.FromBase64String(data.data);
        int w = data.width;
        int h = data.height;

        // Decompress zlib: skip 2-byte zlib header, then deflate
        byte[] rawBytes;
        using (var input = new MemoryStream(compressed, 2, compressed.Length - 2))
        using (var deflate = new DeflateStream(input, CompressionMode.Decompress))
        using (var output = new MemoryStream())
        {
            deflate.CopyTo(output);
            rawBytes = output.ToArray();
        }

        ushort[] decoded = new ushort[w * h];
        Buffer.BlockCopy(rawBytes, 0, decoded, 0, Mathf.Min(rawBytes.Length, decoded.Length * 2));

        lock (_depthLock)
        {
            _depthWidth = w;
            _depthHeight = h;
            _depthData = decoded;
            _depthUpdated = true;
        }

        UpdateDepthVisualization(w, h);

        if (depthImageDisplay != null)
        {
            depthImageDisplay.texture = depthDisplayTexture;
        }
        _depthFrameCount++;
    }

    private void UpdateDepthVisualization(int w, int h)
    {
        if (depthDisplayTexture.width != w || depthDisplayTexture.height != h)
        {
            depthDisplayTexture.Reinitialize(w, h, TextureFormat.RGBA32, false);
        }

        Color32[] colors = new Color32[w * h];
        for (int i = 0; i < w * h; i++)
        {
            byte val = (byte)Mathf.Clamp(_depthData[i] / 2500f * 255f, 0, 255);
            colors[i] = new Color32(val, val, val, 255);
        }

        depthDisplayTexture.SetPixels32(colors);
        depthDisplayTexture.Apply();
    }

    public ushort[] GetRawDepthData(out int width, out int height)
    {
        lock (_depthLock)
        {
            _depthUpdated = false;
            width = _depthWidth;
            height = _depthHeight;
            return _depthData;
        }
    }

    public Texture2D GetColorTexture()
    {
        return colorTexture;
    }
}