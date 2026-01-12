# ONNX Runtime Session Initialization Diagnostics
# ================================================
# 
# Build:
#   docker build -t onnx-diagnostics .
#
# Run (with GPU):
#   docker run --gpus all -it onnx-diagnostics
#
# Run with your model mounted:
#   docker run --gpus all -v /path/to/your/models:/models -it onnx-diagnostics
#
# Run specific test:
#   docker run --gpus all -v /path/to/models:/models onnx-diagnostics python onnx_session_diagnostics.py --model /models/your_model.onnx

# Use NVIDIA CUDA base image with Ubuntu
# Change the tag to match your CUDA version (check with: nvidia-smi)
ARG CUDA_VERSION=12.1.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install Python packages
# Pin versions for reproducibility - update as needed
RUN pip3 install --no-cache-dir \
    numpy>=1.24.0 \
    onnx>=1.14.0 \
    onnxruntime-gpu>=1.16.0 \
    protobuf>=3.20.0

# Create working directory
WORKDIR /app

# Copy diagnostic scripts
COPY check_environment.py .
COPY test_gil_detection.py .
COPY onnx_session_diagnostics.py .
COPY onnx_solutions.py .
COPY CLAUDE.md .

# Create directory for user models
RUN mkdir -p /models

# Set environment variables for better CUDA behavior
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command - run environment check
CMD ["python", "check_environment.py"]
