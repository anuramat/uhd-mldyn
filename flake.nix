{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/release-24.11";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudnnSupoprt = true;
          };
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages (
              python-pkgs: with python-pkgs; [
                pip
                jupyter
                jupytext
                torch-bin
                jupyter-collaboration
              ]
            ))
          ];
          shellHook = ''
            # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
            # See https://pip.pypa.io/en/stable/topics/configuration/
            export PIP_PREFIX=$(realpath -m "''${XDG_CACHE_HOME:-$HOME/.cache}/flake_pip/$(pwd)/")
            export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}''${PYTHONPATH:+:$PYTHONPATH}"
            export PATH="$PIP_PREFIX/bin:''${PATH:+:$PATH}"
            unset SOURCE_DATE_EPOCH
          '';
        };
      }
    );
}
