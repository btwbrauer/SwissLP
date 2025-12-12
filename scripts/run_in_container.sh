#!/usr/bin/env bash
# Run training in a Podman container with ROCm support
#
# Usage:
#   ./scripts/run_in_container.sh                    # Run test training
#   ./scripts/run_in_container.sh optimize           # Run hyperparameter optimization
#   ./scripts/run_in_container.sh shell              # Open interactive shell

set -euo pipefail

# Configuration
IMAGE_NAME="swisslp-rocm"
CONTAINER_NAME="swisslp-training"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# GPU Memory limit (in GB) - leave headroom for Hyprland/display
# Your GPU has 16GB, we limit container to 12GB to leave 4GB for display
GPU_MEMORY_LIMIT_GB=12

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if podman is available
if ! command -v podman &> /dev/null; then
    log_error "Podman is not installed. Please install podman first."
    exit 1
fi

# Check if ROCm device is available
if [ ! -d "/dev/dri" ]; then
    log_warn "/dev/dri not found. GPU might not be available in container."
fi

# Build the container image if it doesn't exist or if Containerfile changed
build_image() {
    log_info "Building container image: $IMAGE_NAME"
    podman build -t "$IMAGE_NAME" -f Containerfile "$PROJECT_DIR"
}

# Run container with GPU access (safer settings to prevent Hyprland crashes)
run_container() {
    local cmd="${1:-}"
    
    log_info "Running container with ROCm GPU access..."
    log_warn "GPU memory limited to ${GPU_MEMORY_LIMIT_GB}GB to prevent display crashes"
    
    # Common podman run options for ROCm with safety limits
    local podman_opts=(
        --rm
        --name "$CONTAINER_NAME"
        # GPU access with render group
        --device=/dev/kfd
        --device=/dev/dri
        --group-add video
        --group-add render
        --security-opt seccomp=unconfined
        # Memory limits to prevent GPU exhaustion
        --memory=32g
        --memory-swap=48g
        # CPU limits to reduce system stress
        --cpus=8
        # Volume mounts
        -v "$PROJECT_DIR/data:/app/data:ro"
        -v "$PROJECT_DIR/configs:/app/configs:ro"
        -v "$PROJECT_DIR/src:/app/src:ro"
        -v "$PROJECT_DIR/scripts:/app/scripts:ro"
        -v "$PROJECT_DIR/outputs:/app/outputs:rw"
        -v "$PROJECT_DIR/mlruns:/app/mlruns:rw"
        # ROCm environment variables
        -e ROCR_VISIBLE_DEVICES=0
        -e HSA_OVERRIDE_GFX_VERSION=12.0.1
        -e HIP_FORCE_DEV_KERNARG=1
        -e ROCM_FORCE_DEV_KERNARG=1
        # GPU memory limit via PyTorch
        -e "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
        -e "GPU_MAX_HEAP_SIZE=${GPU_MEMORY_LIMIT_GB}G"
        -e "GPU_MAX_ALLOC_PERCENT=75"
        # Prevent GPU lockup
        -e "HSA_FORCE_FINE_GRAIN_PCIE=1"
        -e "AMD_SERIALIZE_KERNEL=3"
        -e "AMD_SERIALIZE_COPY=3"
    )
    
    case "$cmd" in
        shell)
            log_info "Starting interactive shell..."
            podman run -it "${podman_opts[@]}" "$IMAGE_NAME" /bin/bash
            ;;
        optimize)
            log_info "Running hyperparameter optimization..."
            podman run "${podman_opts[@]}" "$IMAGE_NAME" \
                python scripts/optimize_models.py --models german_bert --n-trials 60
            ;;
        test)
            log_info "Running test training..."
            podman run "${podman_opts[@]}" "$IMAGE_NAME" \
                python scripts/test_text_models_training.py --model german_bert
            ;;
        versions)
            log_info "Checking versions..."
            podman run "${podman_opts[@]}" "$IMAGE_NAME" \
                python -c "import torch; import transformers; print(f'Torch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'ROCm: {torch.version.hip}')"
            ;;
        gpu-check)
            log_info "Checking GPU status (safe - no compute)..."
            podman run "${podman_opts[@]}" "$IMAGE_NAME" \
                python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
            ;;
        *)
            log_info "Running default test training..."
            podman run "${podman_opts[@]}" "$IMAGE_NAME" \
                python scripts/test_text_models_training.py --model german_bert
            ;;
    esac
}

# Main
main() {
    local action="${1:-test}"
    
    # Always rebuild to ensure latest code
    build_image
    
    # Run the container
    run_container "$action"
}

main "$@"


