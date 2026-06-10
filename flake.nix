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

              # --- MLIR tools: mlir-opt / mlir-translate / mlir-* (used by
              #     rclite/codegen/mlir_*.py and tests/mlir_*_test.py).
              #     Confirmed present in the locked nixpkgs pin as
              #     llvmPackages_20.mlir == 20.1.8 — the SAME LLVM patch as the
              #     llvmlite wheel embeds (llvm.llvm_version_info == (20,1,8)).
              #     rclite.codegen.mlir_jit.tools_available() gates the MLIR
              #     bridge on this major matching llvmlite's, so pinning it here
              #     is what makes the host<->device bit-exact MLIR tests actually
              #     RUN (rather than skip) — see the shellHook guard below. ---
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
              # --- Pin the MLIR toolchain to llvmlite's LLVM, exclude host CIRCT ---
              # llvmlite embeds LLVM 20 (today 20.1.8). The MLIR bridge
              # (mlir-opt -> mlir-translate -> llvmlite, see
              # rclite.codegen.mlir_jit.tools_available) only runs when the CLI
              # tools' LLVM *major* matches llvmlite's. The host's ~/opt/circt
              # ships an LLVM-23 mlir-opt/llc that the version gate refuses (so the
              # bit-exact MLIR tests would silently skip) and that also miscompiles
              # the generic-form IR. Merely prepending isn't enough — actively
              # SCRUB every CIRCT entry out of PATH first, then put the pinned nix
              # LLVM-20 mlir + llvm bins ahead of everything else.
              export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
                | grep -vE '/circt(/|$)|/opt/circt' | paste -sd: -)"
              export PATH="${llvm.mlir}/bin:${llvm.llvm}/bin:$PATH"

              # The AVR cross toolchain's setup hook exports CC=avr-gcc. Rust's
              # cc-rs (build.rs) then uses it to compile HOST objects too, passing
              # -m64 which avr's cc1 rejects (breaks the cargo FFI round-trip
              # tests). Reset CC/CXX to host gcc — the Arduino target calls
              # `avr-gcc` by name, so AVR builds are unaffected.
              export CC=gcc
              export CXX=g++

              # --- Loud guard: never silently vouch for bit-exactness ---
              # If anything but the pinned nix LLVM-20 mlir-opt resolves (e.g.
              # CIRCT leaked back in), say so on stderr instead of letting the
              # tests quietly skip / a mismatched tool slip through.
              _mo="$(command -v mlir-opt || true)"
              _mver="$(mlir-opt --version 2>/dev/null \
                | sed -n 's/.*LLVM version \([0-9][0-9]*\).*/\1/p' | head -1)"
              case "$_mo:$_mver" in
                ${llvm.mlir}/*:20) : ;;  # OK — pinned nix LLVM-20.x
                *) echo "WARNING: mlir-opt is NOT the pinned nix LLVM-20 (resolved" \
                        "'$_mo', LLVM major '$_mver'). CIRCT/host LLVM may have" \
                        "leaked into PATH; MLIR bit-exact tests cannot be trusted." >&2 ;;
              esac

              echo "rclite devShell — MLIR pinned to llvmlite's LLVM 20 (CIRCT excluded)"
              echo "  mlir-opt      : $_mo  [LLVM $_mver]"
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
