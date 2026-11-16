#!/bin/bash
# First step is to pull the docker image from Docker Hub: docker image pull imax11/ros2-jazzy-dev:latest
# or build the dev image yourself: docker build --build-arg USERNAME=devuser -t lasr_grasp:dev -f Dockerfile --target dev .

xhost local:root

SCRIPTPATH="$( cd -- "$(dirname "$0/..")" >/dev/null 2>&1 ; pwd -P )"

# If you got issues with networking or GUI (on native linux), make sure that the docker context is set to default!! (cost me a lot of time)
# docker context ls
# docker context use default

# when running on wsl, --network=host doesn't work, need to forward ports manually (8765 for foxglove)
if [[ $(grep -i Microsoft /proc/version) ]]
then
    docker run -it -p 8765:8765 --ipc=host --privileged \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --env="TERM=xterm-256color" \
        --volume="/etc/timezone:/etc/timezone:ro" \
        --volume="/etc/localtime:/etc/localtime:ro" \
        --volume="$SCRIPTPATH/../../:/home/devuser/lasr_grasp_ws" \
        --volume="$SCRIPTPATH/configs/.tmux.conf:/home/devuver/.tmux.conf" \
        lasr_grasp:dev bash -c "cd ~ && bash"
else
    docker run -it --network=host --ipc=host --privileged\
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/etc/timezone:/etc/timezone:ro" \
        --volume="/etc/localtime:/etc/localtime:ro" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="${XAUTHORITY}:/root/.Xauthority" \
        --env="TERM=xterm-256color" \
        --volume="$SCRIPTPATH/../../:/home/devuser/lasr_grasp_ws" \
        --volume="$SCRIPTPATH/configs/.tmux.conf:/home/devuser/.tmux.conf" \
        --group-add plugdev \
        -v /run/udev:/run/udev:ro \
        -v /dev/bus/usb:/dev/bus/usb \
        -v /dev/snd:/dev/snd \
        -e DISPLAY=unix$DISPLAY \
        --device="/dev/ttyACM1:/dev/ttyACM1" \
        --device="/dev/snd:/dev/snd" \
        --group-add $(getent group dialout | cut -d: -f3) \
        --group-add $(getent group video | cut -d: -f3) \
        lasr_grasp:dev bash -c "cd ~ && bash"
fi
