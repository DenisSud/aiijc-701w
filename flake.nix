{
  description = "PyTorch + CUDA dev shell (Python 3.11, ML, LSP, linters)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      python = pkgs.python311;
      pythonEnv = python.withPackages (ps: with ps; [
        torch
        pandas
        ollama
        pip
        python-lsp-server
        openai
        pytest
        pytest-asyncio
      ]);
    in {
      devShells.default = pkgs.mkShell {
        buildInputs = [
          pythonEnv
          pkgs.cudatoolkit
          pkgs.cudaPackages.cudnn
          pkgs.cudaPackages.cuda_cudart
        ];

        shellHook = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(
            ${pkgs.lib.makeLibraryPath [
              pkgs.cudatoolkit
              pkgs.cudaPackages.cudnn
              "/run/opengl-driver"
            ]}
          )

          # Avoid hangs in asyncified get_platform
          export PYTORCH_NO_CUDA_MEMORY_CACHING=1
          export CUDA_LAUNCH_BLOCKING=1
        '';
      };
    }
  );
}
