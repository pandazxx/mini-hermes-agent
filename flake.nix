{
  description = "mini-swe-agent development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python311
            pkgs.uv
            pkgs.just
          ];

          shellHook = ''
            echo "mini-swe-agent dev shell - python $(python3 --version | cut -d' ' -f2), uv $(uv --version | cut -d' ' -f2)"
            echo "Run 'just install' once, then 'just --list' for available shortcuts."
            for tool in codex claude; do
              command -v "$tool" >/dev/null 2>&1 || echo "note: '$tool' not on PATH (needed by the hermes agent's --tool $tool)"
            done
          '';
        };
      });
}
