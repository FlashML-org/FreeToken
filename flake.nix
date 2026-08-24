{
  description = "FreeToken sm_75 (RTX 2080 Ti) development environment";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree    = true;
            cudaSupport    = true;
          };
        };

        # CUDA 12.8 — maximum supported by the RTX 2080 Ti (driver r553 ceiling).
        # nixos-unstable carries cudaPackages_12_8.
        cuda = pkgs.cudaPackages_12_8;

      in {
        devShells.default = pkgs.mkShell {
          name = "freetoken-sm75";

          packages = [
            # Python tooling
            pkgs.uv
            pkgs.python312

            # CUDA 12.8 toolkit
            cuda.cudatoolkit
            cuda.cudnn
            cuda.nccl          # required for TP4 allreduce across 4 GPUs

            # nvcc 12.8 requires gcc ≤ 12; gcc13+ breaks device-code compilation
            pkgs.gcc12

            # Build tools
            pkgs.cmake
            pkgs.ninja
            pkgs.git
          ];

          env = {
            CUDA_HOME            = "${cuda.cudatoolkit}";
            CUDA_PATH            = "${cuda.cudatoolkit}";
            NCCL_HOME            = "${cuda.nccl}";

            # sm_75 default arch — also picked up by torch.utils.cpp_extension
            # and by setup.py's os.environ.setdefault() (which won't override this).
            TORCH_CUDA_ARCH_LIST = "7.5;8.0;8.6;8.9;9.0";

            # Prefer NVLink for P2P between paired cards (0+1, 2+3)
            NCCL_P2P_LEVEL  = "NVL";
            NCCL_SHM_DISABLE = "0";

            # nvcc 12.8 (CUDA major 12) matches torch cu128 (CUDA major 12),
            # so check_nvcc_matches_torch() passes without this. Kept as a
            # documentation marker and emergency override.
            # FREETOKEN_ALLOW_CUDA_MISMATCH = "1";

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              cuda.cudatoolkit
              cuda.cudnn
              cuda.nccl
              "${pkgs.gcc12.cc.lib}"
            ];
          };

          shellHook = ''
            echo "╔══════════════════════════════════════════════════════════╗"
            echo "║  FreeToken sm_75 dev shell — CUDA 12.8 / RTX 2080 Ti    ║"
            echo "╚══════════════════════════════════════════════════════════╝"
            nvcc_ver=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "not found")
            echo "  nvcc:   $nvcc_ver"
            gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
                       || echo "  nvidia-smi not available (driver passthrough required)")
            echo "  GPUs:"
            echo "$gpu_info" | while IFS= read -r line; do echo "    $line"; done
            echo ""

            # Create / activate venv
            if [ ! -d ".venv" ]; then
              uv venv --python python3.12
            fi
            source .venv/bin/activate
            echo "  venv:   $VIRTUAL_ENV"
            echo ""
          '';
        };
      }
    );
}
