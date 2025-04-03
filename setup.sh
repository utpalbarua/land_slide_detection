#!/bin/bash

# Install system dependencies based on OS
if [ -f /etc/debian_version ]; then
    # Debian/Ubuntu
    echo "Installing dependencies for Debian/Ubuntu..."
    apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
elif [ -f /etc/alpine-release ]; then
    # Alpine
    echo "Installing dependencies for Alpine..."
    apk add --update --no-cache libxcb mesa-gl
elif [ -f /etc/redhat-release ]; then
    # CentOS/RHEL
    echo "Installing dependencies for CentOS/RHEL..."
    yum install -y mesa-libGL
else
    echo "Unsupported OS. Please install libGL.so.1 manually."
    exit 1
fi

# Install Python dependencies
pip install -r requirements.txt

echo "Setup completed successfully!"
