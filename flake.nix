{
  description = "portc: Parallel Out-of-Core Randomized Tensor-Network Compression";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      # x86_64-darwin is not supported by nixpkgs-unstable since 26.11.
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forEachSystem (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.git
            pkgs.gh
            # Builds the documentation site under docs/.
            pkgs.nodejs_22
          ];

          env = {
            # Python itself and all project dependencies stay under uv's control,
            # so the flake only has to provide uv and the native libraries that
            # binary wheels (numpy, quimb, cupy, ...) dlopen at runtime.
            UV_PYTHON_DOWNLOADS = "automatic";
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ];
          };

          # GPU extras are deliberately not synced: they are CUDA/ROCm specific.
          shellHook = ''
            uv sync --all-groups
            # Activate by hand: .venv/bin/activate hardcodes paths that break
            # when the checkout is shared, and uv may place the venv elsewhere.
            export VIRTUAL_ENV="''${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
            export PATH="$VIRTUAL_ENV/bin:$PATH"
          '';
        };
      });

      formatter = forEachSystem (pkgs: pkgs.nixfmt);
    };
}
