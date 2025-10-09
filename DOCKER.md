# Docker Usage

This project provides a Docker container based on the official Nix container with interactive environment selection.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run with interactive selection
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
docker-compose exec swisslp bash
```

### Using Docker directly

```bash
# Build the image
docker build -t swisslp .

# Run with volume mounting
docker run -it -v $(pwd):/workspace swisslp

# Or with specific configuration
docker run -it -v $(pwd):/workspace swisslp bash -c "echo '2' | /entrypoint.sh"
```

## Available Configurations

When you start the container, you'll be prompted to choose:

1. **default** - CPU-only environment (recommended for development)
2. **cuda** - CUDA GPU support (requires NVIDIA GPU and drivers)
3. **rocm** - ROCm GPU support (requires AMD GPU and ROCm drivers)

## Features

- Based on official `nixos/nix` container
- Interactive configuration selection
- Volume mounting for persistent development
- All dependencies from the flake are available
- Hugging Face cache directory setup
- Python path configured for the project

## Notes

- The container mounts the current directory to `/workspace`
- Nix store is persisted in Docker volumes for faster rebuilds
- GPU support requires appropriate host drivers and Docker GPU runtime
