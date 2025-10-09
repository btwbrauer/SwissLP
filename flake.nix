{
  description = "Swiss German Language Processing - Text and Speech Classification";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05/";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Common Python packages (minimal but complete)
        commonPythonDeps = ps: with ps; [
          # Hugging Face ecosystem
          transformers
          datasets
          tokenizers
          accelerate

          # NLP & Audio processing
          nltk
          sentencepiece
          librosa
          soundfile
          torchaudio
          torchvision

          # Scientific Computing
          numpy
          pandas
          scikit-learn

          # Dev Tools
          jupyter
          pytest
          pytest-cov

          # Utilities
          tqdm
          pyyaml
        ];

        # Python environment factory that pins `torch` to a specific variant
        # so transitive deps (e.g. transformers/accelerate) reuse the same torch.
        mkPythonEnv = torchPkg:
          let
            pythonWithPinnedTorch = pkgs.python312.override {
              packageOverrides = self: super: {
                torch = torchPkg;
              };
            };
          in pythonWithPinnedTorch.withPackages (ps:
            commonPythonDeps ps ++ [ ps.torch ]
          );

        # Python environments for different hardware
        pythonDefault = mkPythonEnv pkgs.python312Packages.torch;
        pythonCuda = mkPythonEnv pkgs.python312Packages.torchWithCuda;
        pythonRocm = mkPythonEnv pkgs.python312Packages.torchWithRocm;

        # Shell factory
        mkShell = pythonEnv: name: pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.git
            pkgs.jujutsu
            pkgs.lazyjj
            pkgs.ffmpeg
          ];

          shellHook = ''
            export HF_HOME="''${HF_HOME:-$PWD/.cache/huggingface}"
            export PYTHONPATH="$PWD:$PYTHONPATH"
            export TOKENIZERS_PARALLELISM=true
            mkdir -p "$HF_HOME"

            echo "🇨🇭 Swiss Language Processing (${name})"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✓ Environment ready"
          '';
        };

      in
      {
        devShells = {
          default = mkShell pythonDefault "CPU";
          cuda = mkShell pythonCuda "CUDA";
          rocm = mkShell pythonRocm "ROCm";
        };

        packages.default = pkgs.stdenv.mkDerivation {
          pname = "swisslp";
          version = "0.1.0";
          src = ./.;
          buildInputs = [ pythonDefault ];

          installPhase = ''
            mkdir -p $out/bin
            echo "#!${pkgs.bash}/bin/bash" > $out/bin/swisslp
            echo "${pythonDefault}/bin/python \"\$@\"" >> $out/bin/swisslp
            chmod +x $out/bin/swisslp
          '';
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/swisslp";
        };
      }
    );
}

