# Docker Setup for ONNX Runtime Diagnostics

## Prerequisites

### 1. Docker
```bash
# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### 2. NVIDIA Container Toolkit (for GPU support)
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### Using the Helper Script (Recommended)

```bash
# Make script executable
chmod +x run_docker.sh

# Build the container
./run_docker.sh build

# Run environment check
./run_docker.sh check

# Run GIL detection test
./run_docker.sh gil

# Run full diagnostics with your model
MODEL_PATH=/path/to/your/model.onnx ./run_docker.sh diagnose

# Or start interactive shell
./run_docker.sh shell
```

### Using Docker Directly

```bash
# Build
docker build -t onnx-diagnostics .

# Run environment check
docker run --rm --gpus all onnx-diagnostics

# Run GIL test
docker run --rm --gpus all onnx-diagnostics python test_gil_detection.py

# Run with your model
docker run --rm --gpus all \
    -v /path/to/your/models:/models \
    onnx-diagnostics \
    python onnx_session_diagnostics.py --model /models/your_model.onnx

# Interactive shell
docker run --rm --gpus all -it \
    -v /path/to/your/models:/models \
    onnx-diagnostics bash
```

### Using Docker Compose

```bash
# Run environment check
docker-compose run diagnostics

# Run GIL test
docker-compose run diagnostics python test_gil_detection.py

# Run full diagnostics (set environment variables)
MODELS_DIR=/path/to/models MODEL_PATH=/models/model.onnx docker-compose run full-diagnostics

# Interactive shell
docker-compose run diagnostics bash
```

## Customizing CUDA Version

Check your CUDA version:
```bash
nvidia-smi  # Look for "CUDA Version: XX.X"
```

Build with matching version:
```bash
# Using helper script
CUDA_VERSION=11.8.0 ./run_docker.sh build

# Using docker build directly
docker build --build-arg CUDA_VERSION=11.8.0 -t onnx-diagnostics .

# Using docker-compose (edit docker-compose.yml or use env var)
CUDA_VERSION=11.8.0 docker-compose build
```

### Common CUDA Versions

| Your nvidia-smi shows | Use CUDA_VERSION |
|-----------------------|------------------|
| CUDA Version: 12.x    | 12.1.0 (default) |
| CUDA Version: 11.8    | 11.8.0           |
| CUDA Version: 11.7    | 11.7.1           |

## Troubleshooting

### "could not select device driver" error
```bash
# NVIDIA Container Toolkit not installed or Docker not restarted
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### "no CUDA-capable device is detected"
```bash
# Check if GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If that fails, check host GPU
nvidia-smi
```

### CUDA version mismatch
```bash
# Error like: "CUDA driver version is insufficient for CUDA runtime version"
# Solution: Rebuild with correct CUDA version
nvidia-smi  # Check your version
CUDA_VERSION=11.8.0 ./run_docker.sh build  # Use matching version
```

### Permission denied on run_docker.sh
```bash
chmod +x run_docker.sh
```

### Model file not found in container
```bash
# Make sure to mount the correct directory
docker run --gpus all -v /absolute/path/to/models:/models onnx-diagnostics \
    python onnx_session_diagnostics.py --model /models/your_model.onnx

# Check what's mounted
docker run --gpus all -v /path/to/models:/models onnx-diagnostics ls -la /models
```

## File Structure

```
.
├── Dockerfile              # Container definition
├── docker-compose.yml      # Compose configuration
├── run_docker.sh           # Helper script
├── DOCKER_README.md        # This file
├── CLAUDE.md               # Project context for Claude Code
├── check_environment.py    # Environment verification
├── test_gil_detection.py   # GIL detection test
├── onnx_session_diagnostics.py  # Full diagnostic suite
└── onnx_solutions.py       # Solution implementations
```

## Cloud Usage

### AWS EC2
```bash
# Use Deep Learning AMI (has NVIDIA drivers pre-installed)
# Or install manually on GPU instance (p3, g4dn, etc.)

# SSH to instance, then:
git clone <your-repo> && cd <repo>
./run_docker.sh build
./run_docker.sh check
```

### GCP Compute Engine
```bash
# Use Deep Learning VM or install drivers
# https://cloud.google.com/compute/docs/gpus/install-drivers-gpu

./run_docker.sh build
./run_docker.sh check
```

### Lambda Labs / vast.ai / RunPod
```bash
# Usually have Docker + NVIDIA ready
# Just clone and run
./run_docker.sh build
./run_docker.sh check
```
