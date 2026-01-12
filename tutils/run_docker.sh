#!/bin/bash
# =============================================================================
# ONNX Runtime Diagnostics - Docker Runner
# =============================================================================
#
# Usage:
#   ./run_docker.sh check          # Check environment
#   ./run_docker.sh gil            # Run GIL detection test
#   ./run_docker.sh diagnose       # Run full diagnostics (needs MODEL_PATH)
#   ./run_docker.sh shell          # Interactive shell
#   ./run_docker.sh build          # Build/rebuild container
#
# Environment variables:
#   MODEL_PATH    - Path to your ONNX model file
#   MODELS_DIR    - Directory containing models (mounted to /models)
#   OUTPUT_DIR    - Directory for output files (mounted to /output)
#   NUM_GPUS      - Number of GPUs to test (default: auto-detect)
#   CUDA_VERSION  - CUDA version for build (default: 12.1.0)
#
# Examples:
#   MODEL_PATH=/path/to/model.onnx ./run_docker.sh diagnose
#   MODELS_DIR=/data/models ./run_docker.sh shell

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODELS_DIR="${MODELS_DIR:-./models}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
IMAGE_NAME="onnx-diagnostics"

# Create directories if they don't exist
mkdir -p "$MODELS_DIR" "$OUTPUT_DIR"

# Check for NVIDIA Docker support
check_nvidia_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        echo -e "${YELLOW}Warning: NVIDIA Container Toolkit may not be installed${NC}"
        echo "GPU tests may not work. Install from:"
        echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
    fi
}

# Build the container
build_container() {
    echo -e "${GREEN}Building Docker container...${NC}"
    
    # Check CUDA version on host
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        echo "Detected NVIDIA driver: $DETECTED_CUDA"
    fi
    
    CUDA_VERSION="${CUDA_VERSION:-12.1.0}"
    echo "Building with CUDA version: $CUDA_VERSION"
    
    docker build \
        --build-arg CUDA_VERSION="$CUDA_VERSION" \
        --build-arg CUDNN_VERSION="${CUDNN_VERSION:-8}" \
        --build-arg UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}" \
        -t "$IMAGE_NAME" .
    
    echo -e "${GREEN}Build complete!${NC}"
}

# Run container with GPU support
run_container() {
    local cmd="$@"
    
    docker run --rm \
        --gpus all \
        -v "$MODELS_DIR:/models:ro" \
        -v "$OUTPUT_DIR:/output" \
        -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}" \
        "$IMAGE_NAME" \
        $cmd
}

# Run interactive shell
run_shell() {
    echo -e "${GREEN}Starting interactive shell...${NC}"
    echo "Models mounted at: /models"
    echo "Output directory: /output"
    echo ""
    
    docker run --rm -it \
        --gpus all \
        -v "$MODELS_DIR:/models:ro" \
        -v "$OUTPUT_DIR:/output" \
        -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}" \
        "$IMAGE_NAME" \
        bash
}

# Main command handler
case "${1:-help}" in
    build)
        check_nvidia_docker
        build_container
        ;;
    
    check|env)
        check_nvidia_docker
        echo -e "${GREEN}Running environment check...${NC}"
        run_container python check_environment.py
        ;;
    
    gil|gil-test)
        check_nvidia_docker
        echo -e "${GREEN}Running GIL detection test...${NC}"
        run_container python test_gil_detection.py
        ;;
    
    diagnose|diagnostics|full)
        check_nvidia_docker
        
        if [ -z "$MODEL_PATH" ]; then
            echo -e "${YELLOW}No MODEL_PATH specified. Using --create-dummy${NC}"
            run_container python onnx_session_diagnostics.py --create-dummy
        else
            # Check if model exists
            if [ ! -f "$MODEL_PATH" ]; then
                echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
                exit 1
            fi
            
            # Get the model filename
            MODEL_FILENAME=$(basename "$MODEL_PATH")
            MODEL_DIRNAME=$(dirname "$MODEL_PATH")
            
            echo -e "${GREEN}Running full diagnostics on: $MODEL_PATH${NC}"
            
            # Mount the model's directory and run
            docker run --rm \
                --gpus all \
                -v "$MODEL_DIRNAME:/model_dir:ro" \
                -v "$OUTPUT_DIR:/output" \
                -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}" \
                "$IMAGE_NAME" \
                python onnx_session_diagnostics.py \
                    --model "/model_dir/$MODEL_FILENAME" \
                    ${NUM_GPUS:+--num-gpus $NUM_GPUS}
        fi
        ;;
    
    shell|bash|interactive)
        check_nvidia_docker
        run_shell
        ;;
    
    help|--help|-h|*)
        echo "ONNX Runtime Diagnostics - Docker Runner"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  build       Build the Docker container"
        echo "  check       Run environment check"
        echo "  gil         Run GIL detection test"
        echo "  diagnose    Run full diagnostics (set MODEL_PATH env var)"
        echo "  shell       Start interactive shell"
        echo "  help        Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  MODEL_PATH      Path to ONNX model for diagnostics"
        echo "  MODELS_DIR      Directory to mount at /models (default: ./models)"
        echo "  OUTPUT_DIR      Directory for output files (default: ./output)"
        echo "  NUM_GPUS        Number of GPUs to test"
        echo "  CUDA_VERSION    CUDA version for build (default: 12.1.0)"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 check"
        echo "  $0 gil"
        echo "  MODEL_PATH=/data/model.onnx $0 diagnose"
        echo "  MODELS_DIR=/data/models $0 shell"
        ;;
esac
