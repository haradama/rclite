{
  # Hybrid toolchain pin for rclite (option B):
  #   * Python stays on uv  (pyproject.toml + uv.lock) — unchanged.
  #   * This flake's devShell pins ONLY the native cross-toolchain layer that
  #     was previously scattered across apt / linuxbrew / ~/opt/circt / ~/.cargo
  #     / ~/.wasmtime — the part that actually hurts reproducibility.
  #
  # The concrete bug this fixes: on the host, `llc` / `mlir-opt` /
  # `mlir-translate` resolve to ~/opt/circt/bin (CIRCT's bundled LLVM), whose
  # `llc` lacks the ARM/thumb targets — so cross-compiling to thumbv6m
  # (Cortex-M0) / thumbv4t (GBA) / wasm32 fails. The shellHook below puts the
  # LLVM-20 build (all targets registered) first on PATH, deterministically.
  #
  # Deliberately OUT of scope (manage separately — awkward/absent in nixpkgs):
  #   devkitARM (GBA), llvm-mos (NES 6502), Mesen2, arduino-cli + AVR cores.

  description = "rclite — native cross-toolchain devShell (LLVM 20 + cross + emulators); Python via uv";

  # LLVM 20 lives on nixos-unstable (the 24.11 release tops out at LLVM 19).
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          llvm = pkgs.llvmPackages_20;
        in {
          default = pkgs.mkShell {
            name = "rclite-dev";

            packages = with pkgs; [
              # --- host build / link (memory: host link is gcc, NOT clang) ---
              gcc
              gnumake
              pkg-config

              # --- Python env manager (reproducible uv; manages its own CPython) ---
              uv

              # --- LLVM 20: llc / opt WITH all targets registered. This is the
              #     fix for the CIRCT `llc` collision on PATH. ---
              llvm.llvm
              llvm.lld

              # --- MLIR tools: mlir-opt / mlir-translate (used by rclite/codegen
              #     mlir_*.py and tests/mlir_*_test.py).
              #     NOTE: this is the ONE attribute to verify against your nixpkgs
              #     pin. If `nix develop` errors that `llvmPackages_20.mlir` is
              #     missing, either (a) bump the nixpkgs input to a channel that
              #     ships it, or (b) comment this line out — the rest of the shell
              #     (the llc fix) still works without it. ---
              llvm.mlir

              # --- emulators / runners exercised by tests + benchmarks ---
              qemu      # qemu-system-arm — Cortex-M0 (-icount)
              wasmtime  # wasm32 runner
              simavr    # AVR / Arduino Uno bit-exact parity

              # --- AVR cross compiler (avr-gcc). Built from source unless cached;
              #     swap for your preferred AVR toolchain attr if you have one. ---
              pkgsCross.avr.buildPackages.gcc
            ];

            shellHook = ''
              # Force nix LLVM-20 binaries ahead of ~/opt/circt and host tools.
              # Keep both mlir and llvm bins explicit so mlir-opt/mlir-translate
              # and llc are guaranteed to come from the same LLVM major.
              export PATH="${llvm.mlir}/bin:${llvm.llvm}/bin:$PATH"

              # The AVR cross toolchain's setup hook exports CC=avr-gcc. Rust's
              # cc-rs (build.rs) then uses it to compile HOST objects too, passing
              # -m64 which avr's cc1 rejects (breaks the cargo FFI round-trip
              # tests). Reset CC/CXX to host gcc — the Arduino target calls
              # `avr-gcc` by name, so AVR builds are unaffected.
              export CC=gcc
              export CXX=g++
              echo "rclite devShell"
              echo "  mlir-opt      : $(command -v mlir-opt)"
              echo "  mlir-translate: $(command -v mlir-translate)"
              echo "  llc           : $(command -v llc)"
              echo "  llc targets   : $(llc --version 2>/dev/null | grep -A1 'Registered Targets' >/dev/null && echo ok || echo '??')"
              echo "  Python        : uv ($(uv --version 2>/dev/null)) — run 'uv sync' then 'uv run pytest'"
              echo "  Out of scope  : devkitARM(GBA), llvm-mos(NES), Mesen2, arduino-cli — manage separately."
            '';
          };
        });
    };
}
