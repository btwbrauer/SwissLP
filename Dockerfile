FROM nixos/nix:latest

# Install additional tools (git is already available as git-minimal)
RUN nix-env -iA nixpkgs.bash nixpkgs.coreutils

# Set working directory
WORKDIR /workspace

# Copy the flake files
COPY flake.nix flake.lock ./

# Enable experimental Nix features
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf

# Create entrypoint script
RUN cat > /entrypoint.sh << 'EOF'
#!/usr/bin/env bash

echo "🇨🇭 Swiss Language Processing - Docker Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Available configurations:"
echo "1) default (CPU only)"
echo "2) cuda (CUDA GPU support)"
echo "3) rocm (ROCm GPU support)"
echo ""
read -p "Select configuration [1-3, default: 1]: " choice

case ${choice:-1} in
    1|default)
        echo "Starting default (CPU) environment..."
        nix develop .#default
        ;;
    2|cuda)
        echo "Starting CUDA environment..."
        nix develop .#cuda
        ;;
    3|rocm)
        echo "Starting ROCm environment..."
        nix develop .#rocm
        ;;
    *)
        echo "Invalid choice, starting default environment..."
        nix develop .#default
        ;;
esac
EOF

RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["bash"]
