{
  description = "PyTorch + CUDA dev shell (Python 3.13, ML, LSP, linters)";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = pkgs.python311;
        pythonEnv = python.withPackages (ps: with ps; [
          pytorch-bin
          transformers
          numpy
          pandas
          seaborn
          jupyter
          pip
          black
          mypy
          python-lsp-server
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
          '';
        };
      }
    );
}
