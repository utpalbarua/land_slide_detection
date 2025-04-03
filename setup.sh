#!/bin/bash

# Check for Python installation and install if necessary
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Installing Python3..."
    sudo apt update && sudo apt install -y python3 python3-pip
else
    echo "Python3 is already installed."
fi

# Install system dependencies based on OS
if [ -f /etc/debian_version ]; then
    # Debian/Ubuntu
    echo "Installing dependencies for Debian/Ubuntu..."
    sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
elif [ -f /etc/alpine-release ]; then
    # Alpine
    echo "Installing dependencies for Alpine..."
    sudo apk add --update --no-cache libxcb mesa-gl
elif [ -f /etc/redhat-release ]; then
    # CentOS/RHEL
    echo "Installing dependencies for CentOS/RHEL..."
    sudo yum install -y mesa-libGL
else
    echo "Unsupported OS. Please install libGL.so.1 manually."
    exit 1
fi

# Ensure pip is installed
if ! command -v pip3 &>/dev/null; then
    echo "pip3 is not installed. Installing pip3..."
    sudo apt install -y python3-pip
else
    echo "pip3 is already installed."
fi

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip3 install -r requirements.txt

# Install FFmpeg
echo "Installing FFmpeg..."
sudo apt update && sudo apt install -y ffmpeg libavcodec-extra

# Confirm installation
if command -v ffmpeg &>/dev/null; then
    echo "FFmpeg has been installed successfully."
else
    echo "FFmpeg installation failed. Please check the logs."
    exit 1
fi

echo "Setup completed successfully!"
