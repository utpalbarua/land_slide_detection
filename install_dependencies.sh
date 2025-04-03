#!/bin/bash

# Check for Python installation and install if necessary
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Installing Python3..."
    sudo apt update && sudo apt install -y python3 python3-pip
else
    echo "Python3 is already installed."
fi

# Ensure pip is installed
if ! command -v pip3 &>/dev/null; then
    echo "pip3 is not installed. Installing pip3..."
    sudo apt install -y python3-pip
else
    echo "pip3 is already installed."
fi

# Install system dependencies for OpenCV and FFmpeg
echo "Installing system dependencies for OpenCV and FFmpeg..."
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
sudo apt-get install -y ffmpeg libavcodec-extra

# Install OpenCV with FFmpeg support
echo "Installing OpenCV with FFmpeg support..."
pip3 install opencv-python-headless

# Install YOLO and ultralytics
echo "Installing ultralytics (YOLO)..."
pip3 install ultralytics

# Install Python dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip3 install -r requirements.txt
else
    echo "requirements.txt not found. Skipping Python dependencies installation."
fi

# Verify FFmpeg installation
if command -v ffmpeg &>/dev/null; then
    echo "FFmpeg has been installed successfully."
else
    echo "FFmpeg installation failed. Please check the logs."
    exit 1
fi

# Confirmation of completion
echo "Setup completed successfully!"

# Final check for Python packages
echo "Verifying installations..."
python3 -c "import cv2; import ultralytics; print('OpenCV version:', cv2.__version__); print('Ultralytics package is installed successfully.')"

echo "Installation process complete!"
