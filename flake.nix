{
  description = "Swiss German Language Processing - Text and Speech Classification";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Common Python packages
        commonPythonDeps = ps: with ps; [
          transformers
          datasets
          tokenizers
          accelerate
          nltk
          sentencepiece
          librosa
          torchaudio
          soundfile  # For audio file I/O (used by torchaudio.save() as fallback)
          torchvision
          numpy
          pandas
          scikit-learn
          matplotlib
          seaborn
          jupyter
          pytest
          tqdm
          mlflow
          optuna
          optuna-dashboard  # Optional: Web UI for Optuna studies
        ];

        # Python environment factory that pins torch at interpreter level
        # This ensures all packages (torchaudio, torchvision, transformers, etc.)
        # use the same torch variant via packageOverrides
        mkPythonEnv = torchPkg:
          let
            pythonWithPinnedTorch = pkgs.python313.override {
              packageOverrides = self: super: {
                torch = torchPkg;
              };
            };
          in
          pythonWithPinnedTorch.withPackages (ps:
            commonPythonDeps ps ++ [ ps.torch ]
          );

        # Python environments for different hardware
        pythonDefault = mkPythonEnv pkgs.python313Packages.torch;
        pythonCuda = mkPythonEnv pkgs.python313Packages.torchWithCuda;
        pythonRocm = mkPythonEnv pkgs.python313Packages.torchWithRocm;

        # Shell factory
        mkShell = pythonEnv: name: pkgs.mkShell {
          packages = with pkgs; [
            pythonEnv
            pyright
            ruff
          ];

          shellHook = ''
            export HF_HOME="''${HF_HOME:-$PWD/.cache/huggingface}"
            export TOKENIZERS_PARALLELISM=true
            export ROCR_VISIBLE_DEVICES=0
            mkdir -p "$HF_HOME"
            
            # Explicitly set Python path for Pyright/editor integration
            # This ensures editors can find the Python interpreter even when direnv
            # doesn't load the environment in non-interactive contexts
            PYTHON_PATH=$(which python 2>/dev/null || echo "${pythonEnv}/bin/python")
            export PYTHON_PATH
            
            # Ensure Python is in PATH for Pyright/editor integration
            # pythonEnv automatically adds Python to PATH via Nix, but we verify it
            if ! command -v python >/dev/null 2>&1; then
              echo "Warning: Python not found in PATH"
            fi
            
            echo "🇨🇭 Swiss Language Processing (${name})"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✓ Environment ready"
            python -m ipykernel install --user --name SwissLP --display-name "SwissLP" 2>&1 || true
          '';
        };
      in
      {
        devShells = {
          default = mkShell pythonRocm "ROCm";
          cuda = mkShell pythonCuda "CUDA";
          cpu = mkShell pythonDefault "CPU";
        };
      }
    );
}
