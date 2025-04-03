#!/bin/bash

# Check for Python 3 and install if necessary
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Installing Python3..."
    sudo apt update && sudo apt install -y python3 python3-pip
else
    echo "Python3 is already installed."
fi

# Check for pip3 and install if necessary
if ! command -v pip3 &>/dev/null; then
    echo "pip3 is not installed. Installing pip3..."
    sudo apt install -y python3-pip
else
    echo "pip3 is already installed."
fi

# Install system dependencies
echo "Installing system dependencies..."
if [ -f /etc/debian_version ]; then
    # Debian/Ubuntu
    sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
elif [ -f /etc/alpine-release ]; then
    # Alpine
    sudo apk add --update --no-cache libxcb mesa-gl
elif [ -f /etc/redhat-release ]; then
    # CentOS/RHEL
    sudo yum install -y mesa-libGL
else
    echo "Unsupported OS. Please install libGL.so.1 manually."
    exit 1
fi

# Install Python dependencies from requirements.txt (if it exists)
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip3 install -r requirements.txt
else
    echo "No requirements.txt found, skipping Python dependency installation."
fi

# Install OpenCV
echo "Installing OpenCV..."
pip3 install opencv-python

# Install ultralytics package
echo "Installing ultralytics..."
pip3 install ultralytics

# Install FFmpeg
echo "Installing FFmpeg..."
sudo apt update && sudo apt install -y ffmpeg libavcodec-extra

# Confirm installation of FFmpeg
if command -v ffmpeg &>/dev/null; then
    echo "FFmpeg has been installed successfully."
else
    echo "FFmpeg installation failed. Please check the logs."
    exit 1
fi

# Confirm OpenCV installation
echo "Verifying OpenCV installation..."
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Confirm ultralytics installation
echo "Verifying ultralytics installation..."
python3 -c "from ultralytics import YOLO; print('Ultralytics version:', YOLO.__version__)"

echo "Setup completed successfully!"
