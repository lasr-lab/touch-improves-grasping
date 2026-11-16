# ROS2 base image
# FROM ros:jazzy AS base
FROM rwthika/ros2-tensorrt:jazzy-ros-base-v25.08 AS base
RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu
RUN ln -s /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi
RUN dpkg-reconfigure libc-bin 

RUN apt update && apt upgrade -y
# @sha256:f39d671d65cb1e7fb7add336579d6e9a3ac0d35ffcc822f598977a580ead9366 

# Install dependencies with apt
RUN apt update && \
  DEBIAN_FRONTEND=noninteractive apt install -y keyboard-configuration && \
  apt install -y git \
  apt-utils \
  software-properties-common \
  bash-completion \
  vim nano tmux htop \
  gdb mold \
  desktop-file-utils \
  ros-dev-tools ros-jazzy-foxglove-bridge\
  python3-colcon-common-extensions \
  python3-pip \
  ros-jazzy-foxglove-bridge \
  ros-jazzy-cv-bridge \
  ros-jazzy-vision-opencv \
  ros-jazzy-ffmpeg-image-transport \
  ros-jazzy-diagnostic-updater \
  ros-jazzy-realsense2-camera \
  ros-jazzy-moveit \
  curl \
  lsb-release \
  gnupg \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-gz-tools-vendor \
  ros-jazzy-gz-sim-vendor \
  ros-jazzy-ros-gz-bridge \
  ipython3 \
  usbutils \
  libportaudio2 \
  alsa-utils \
  alsa-base \
  alsa-tools \
  alsa-firmware-loaders \
  build-essential \
  kmod
  
  
RUN pip install --break-system-packages mypy

FROM base AS dev

# Dev container arguments
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN pip install --break-system-packages colcon-clean pynput
RUN pip install --break-system-packages numpy blinker
RUN pip install --break-system-packages torch
RUN pip install --break-system-packages attrdict omegaconf
RUN pip install --break-system-packages betterproto serial cobs
RUN pip install --break-system-packages einops h5py hdf5plugin torchvision dm_control
RUN pip install --break-system-packages sounddevice pyserial

# Install gazeebo
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt-get update
RUN apt-get install gz-jetty -y

# RUN usermod -d /home/$USERNAME -m $USERNAME
RUN adduser $USERNAME
RUN adduser $USERNAME sudo

RUN cd /home/"$USERNAME"/ && mkdir -p xarm/src && cd xarm/src/ && git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive -b $ROS_DISTRO
RUN cd /home/"$USERNAME"/xarm/src/xarm_ros2/ && git submodule update --init --recursive && git pull --recurse-submodules
RUN cd /home/"$USERNAME"/xarm/src/ && rosdep update
RUN cd /home/"$USERNAME"/xarm/ && /bin/bash -c "source /opt/ros/jazzy/setup.bash; colcon build"
RUN echo "source /home/$USERNAME/xarm/install/setup.bash" >> /home/"$USERNAME"/.bashrc

RUN apt-get install libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev build-essential libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at v4l-utils -y

RUN cd /home/"$USERNAME"/ && git clone https://github.com/realsenseai/librealsense.git && cd librealsense && mkdir build && cd build && cmake .. && cmake --build .

# make sudo available
RUN echo "$USERNAME" ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/"$USERNAME" && \
chmod 0440 /etc/sudoers.d/"$USERNAME"
RUN chown -R ${USERNAME}:${GID} /home/${USERNAME}

# run this one without password
RUN echo "$USERNAME ALL=(ALL) NOPASSWD: /usr/sbin/smartctl" >> /etc/sudoers

# Set the user and source entrypoint in the user's .bashrc file
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/$USERNAME/.bashrc
RUN echo "source /home/$USERNAME/lasr_grasp_ws/install/setup.bash" >> /home/"$USERNAME"/.bashrc
RUN echo "export COLCON_DEFAULTS_FILE=/home/devuser/lasr_grasp_ws/src/configs/colcon_config.yaml" >> /home/"$USERNAME"/.bashrc
RUN chsh -s /bin/bash $USERNAME
ENV SHELL=/bin/bash
USER $USERNAME

# Install colcon mixins 
RUN cd /home/"$USERNAME"/ && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml &&\
colcon mixin update default

RUN pip install --break-system-packages open3d-cpu opencv-python digit_interface
RUN pip uninstall numpy --break-system-packages -y
RUN pip install "numpy<2.0" --break-system-packages

CMD ["/bin/bash"]