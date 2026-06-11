"""Ship an IWRAM-optimized GBA (ARM7TDMI) kernel object plus a C header.

The Game Boy Advance is ARMv4T — **no SIMD** — so the vectorized `.o` route
(`export_optimized_object`) brings no speedup here. But unlike the AVR, the GBA
is a 32-bit ARM core with a mature LLVM backend, and it has a decisive
*non-SIMD* optimization lever: **run the kernel as ARM code from IWRAM** (32 KB,
0-waitstate, 32-bit bus) instead of Thumb code from cartridge ROM (16-bit bus
with waitstates). Measured on the cycle-accurate mGBA, an N=64 i8 reservoir runs

    Thumb in ROM   54,794,838 cycles      (the naive default)
    ARM in IWRAM    6,839,985 cycles      -> 8.0x faster, bit-exact

`export_gba_object` builds the kernel for that contract:

    export_gba_object(qmodel, mode="arm-iwram", out_dir="build/")
      -> build/rc_kernel.o      ARM code + weights in a self-describing
                                `.iwram.rc.*` section (route it into IWRAM)
         build/rc_kernel.h       rc_run(T, X, Y) decl (memref ABI hidden)
         build/rc_kernel.iwram.ld linker fragment + the ROM->IWRAM copy recipe
         build/README.md         how to wire it into a GBA project

`mode="thumb-rom"` instead emits compact Thumb code that runs in place from ROM
(no IWRAM budget needed) — smaller, ~8x slower. Both are bit-exact with the
Python executor (verified on mGBA in `tests/gba_object_test.py`).
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from .info import KernelInfo, info_from_affine
from ..codegen import mlir_jit

_TRIPLES = {
    "arm-iwram": "armv4t-none-eabi",  # ARM instruction set (for IWRAM)
    "thumb-rom": "thumbv4t-none-eabi",  # Thumb (compact, runs from ROM)
}
_CPU = "arm7tdmi"
# objcopy renames the kernel's code/rodata to this self-describing section so a
# one-line `*(.iwram.rc.*)` in the caller's linker script routes it to IWRAM.
_IWRAM_SECTION = ".iwram.rc"

_LINKER_FRAGMENT = """\
/* rclite GBA kernel — IWRAM placement fragment (mode=arm-iwram).
 *
 * 1. Paste the `.rc_iwram` output section below into your GBA linker SECTIONS,
 *    just after your ROM `.text` (load image lives in ROM, run address in IWRAM).
 * 2. Copy the image ROM->IWRAM once at boot (in crt0 or early main), like .data:
 *
 *      extern char __rc_iwram_lma, __rc_iwram_start, __rc_iwram_end;
 *      for (char *d = &__rc_iwram_start, *s = &__rc_iwram_lma;
 *           d < &__rc_iwram_end; ) *d++ = *s++;
 *
 * 3. Then call rc_run(T, X, Y) — the kernel executes from 0-waitstate IWRAM as
 *    ARM code (~8x faster than Thumb-from-ROM on real GBA timing).
 */
    .rc_iwram : {
        . = ALIGN(4);
        __rc_iwram_start = .;
        *(.iwram.rc.text .iwram.rc.rodata)
        . = ALIGN(4);
        __rc_iwram_end = .;
    } > IWRAM AT > ROM
    __rc_iwram_lma = LOADADDR(.rc_iwram);
"""


@dataclass
class GbaObjectBundle:
    """A GBA-targeted kernel object + header (+ IWRAM linker fragment)."""

    name: str
    info: KernelInfo
    mode: str
    object_code: bytes
    header: str
    readme: str
    linker_fragment: str | None = None
    func_name: str = "rc_run"

    def write(self, out_dir) -> pathlib.Path:
        """Write `{name}.o`, `{name}.h`, README, and (arm-iwram) the fragment."""
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.name}.o").write_bytes(self.object_code)
        (out / f"{self.name}.h").write_text(self.header)
        (out / "README.md").write_text(self.readme)
        if self.linker_fragment is not None:
            (out / f"{self.name}.iwram.ld").write_text(self.linker_fragment)
        return out


def _readme(name: str, info: KernelInfo, mode: str) -> str:
    if mode == "arm-iwram":
        place = (
            "ARM code + weights in a `.iwram.rc.*` section — route it to **IWRAM**\n"
            f"using `{name}.iwram.ld` (paste the section, copy ROM->IWRAM at boot)."
        )
        speed = (
            "~8x faster than Thumb-from-ROM on real GBA timing (mGBA-measured)"
        )
    else:
        place = "Thumb code; runs in place from cartridge **ROM** (no IWRAM budget)."
        speed = "compact; ~8x slower than the arm-iwram mode"
    return f"""# {name} — GBA (ARM7TDMI) rclite kernel  [mode: {mode}]

{place}

The integer kernel is **bit-exact** with rclite's Python executor (ARMv4T has no
SIMD, so this is scalar — the speed comes from ARM-vs-Thumb + IWRAM placement,
not vectorization). {speed}.

## Shape
- input  `RC_K = {info.K}` (`rc_in_t` = `{info.storage_ctype}`)
- output `RC_M = {info.M}` (`rc_out_t` = `{info.out_ctype}`)
- reservoir `N = {info.N}` ({info.topology}), head `{info.head}`

## Use
```c
#include "{name}.h"
int8_t Y[T * RC_M];
rc_run(T, X, Y);          /* X: T*RC_K inputs (row-major) */
```
Link with the GBA toolchain (`arm-none-eabi-gcc -mthumb -mthumb-interwork`).
{"For arm-iwram, add `" + name + ".iwram.ld` to your linker script and copy the section to IWRAM at boot (see the fragment)." if mode == "arm-iwram" else ""}

For a quick compact build with no linker work, use `mode="thumb-rom"`.
"""


def export_gba_object(
    qmodel,
    *,
    mode: str = "arm-iwram",
    name: str = "rc_kernel",
    head=None,
    sparse=None,
    out_dir=None,
    objcopy: str = "arm-none-eabi-objcopy",
) -> GbaObjectBundle:
    """Compile `qmodel` to a GBA kernel object for `mode`.

    `qmodel` is an `AffineQuantizedModel`. `mode` is `"arm-iwram"` (ARM code for
    IWRAM, ~8x faster — the default) or `"thumb-rom"` (compact Thumb, runs from
    ROM). `head`/`sparse` select the readout head / CSR-sparse W_res. When
    `out_dir` is given the bundle is also written there.

    The arm-iwram mode renames the kernel's sections (needs `objcopy` on PATH) so
    a one-line `*(.iwram.rc.*)` routes it to IWRAM. Raises if `mode` is unknown.
    """
    from ..codegen.mlir_affine_xdsl import (
        emit_affine_mlir_xdsl,
    )  # optional dep

    if mode not in _TRIPLES:
        raise ValueError(f"mode must be one of {list(_TRIPLES)}, got {mode!r}")
    info = info_from_affine(qmodel, name=name, head=head)
    mlir = emit_affine_mlir_xdsl(qmodel, head=head, sparse=sparse, vlen=1)
    obj = mlir_jit.cross_compile_object(
        mlir, triple=_TRIPLES[mode], cpu=_CPU, filetype="obj"
    )

    fragment = None
    if mode == "arm-iwram":
        if shutil.which(objcopy) is None:
            raise RuntimeError(
                f"mode='arm-iwram' needs {objcopy!r} on PATH (GBA toolchain) to "
                "place the kernel in IWRAM; use mode='thumb-rom' otherwise"
            )
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            (td / "k.o").write_bytes(obj)
            r = subprocess.run(
                [
                    objcopy,
                    "--rename-section",
                    f".text={_IWRAM_SECTION}.text",
                    "--rename-section",
                    f".rodata={_IWRAM_SECTION}.rodata",
                    str(td / "k.o"),
                    str(td / "k2.o"),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"{objcopy} failed:\n{r.stderr[:1500]}")
            obj = (td / "k2.o").read_bytes()
        fragment = _LINKER_FRAGMENT

    header = mlir_jit.emit_c_header(
        K=qmodel.K,
        M=qmodel.M,
        storage_bits=qmodel.storage_bits,
        classify=(head == "classify"),
    )
    bundle = GbaObjectBundle(
        name=name,
        info=info,
        mode=mode,
        object_code=obj,
        header=header,
        readme=_readme(name, info, mode),
        linker_fragment=fragment,
    )
    if out_dir is not None:
        bundle.write(out_dir)
    return bundle
