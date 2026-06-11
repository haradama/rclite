"""Ship a FLASH-RESIDENT AVR kernel object plus a C header.

The SIMD `.o` route (`export_optimized_object`) does not fit AVR: an 8-bit
ATmega has no SIMD (no speed to gain) and — fatally — the MLIR/LLVM object puts
its weight tables in `.rodata`, which on AVR is copied into the 2 KB SRAM at
boot, so any real reservoir overflows. `export_avr_object` takes the route that
actually works on AVR: it compiles rclite's portable integer kernel with
**avr-gcc**, where the weight tables carry `PROGMEM` and therefore live in the
32 KB **Flash** (read with `LPM`), and loop indices stay 16-bit. A real
reservoir fits, and the object is bit-exact with the Python executor.

    export_avr_object(qmodel, mcu="atmega328p", out_dir="build/")
      -> build/rc_kernel.o   avr-gcc object; weights in Flash (PROGMEM/LPM)
         build/rc_kernel.h   dims + float<->quant helpers + rc_predict decl
         build/README.md     how to link it into an avr-gcc / Arduino project

Verified end-to-end on the emulated ATmega328P (simavr): see
`tests/avr_object_test.py`, which links the object into a firmware, runs it, and
checks the UART output matches the executor byte-for-byte.

There is no optimization advantage over shipping the C source (`export_bundle`)
— avr-gcc compiles both identically; the `.o` form is for shipping a binary blob
(hidden source) that an avr-gcc/PlatformIO project links directly.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from .info import KernelInfo, info_from_affine
from .c_header import emit_c_header


@dataclass
class AvrObjectBundle:
    """An avr-gcc-compiled, Flash-resident kernel object + its C header."""

    name: str
    info: KernelInfo
    mcu: str
    object_code: bytes
    header: str
    readme: str
    func_name: str = "rc_predict"

    def write(self, out_dir) -> pathlib.Path:
        """Write `{name}.o`, `{name}.h`, and `README.md` into `out_dir`."""
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.name}.o").write_bytes(self.object_code)
        (out / f"{self.name}.h").write_text(self.header)
        (out / "README.md").write_text(self.readme)
        return out


def _readme(name: str, info: KernelInfo, mcu: str, opt: str) -> str:
    return f"""# {name} — Flash-resident rclite kernel for AVR

`{name}.o` is the reservoir kernel compiled by rclite with **avr-gcc**
(`-mmcu={mcu} {opt}`). The weight/LUT tables carry `PROGMEM`, so they live in
**Flash** (read via `LPM`) — only the small runtime state touches the 2 KB SRAM.
The kernel is **bit-exact** with rclite's Python executor (verified on simavr).

## Shape

- input dim  `RC_K = {info.K}`  (`rc_storage_t` = `{info.storage_ctype}`)
- output dim `RC_M = {info.M}`  (`{info.out_ctype}`)
- reservoir  `N = {info.N}` ({info.topology}), head `{info.head}`

## Use (avr-gcc / PlatformIO)

```c
#include "{name}.h"

int8_t Y[T * RC_M];
rc_predict(T, X, Y);   /* X: T*RC_K inputs (row-major) */
```

```sh
avr-gcc -mmcu={mcu} -c my_app.c -o my_app.o
avr-gcc -mmcu={mcu} my_app.o {name}.o -o firmware.elf
```

For the Arduino IDE, drop `{name}.o` into a *precompiled* library
(`library/src/{mcu}/lib{name}.a`) or just use the source bundle from
`export_bundle` (the Arduino toolchain compiles it to the same code).

Note: AVR has no SIMD, so this object is functionally identical to compiling the
C source; the `.o` form only hides the source. For SIMD targets (x86/NEON/
wasm/RVV) use `export_optimized_object`.
"""


def export_avr_object(
    qmodel,
    *,
    mcu: str = "atmega328p",
    f_cpu: int = 16_000_000,
    opt: str = "-Os",
    name: str = "rc_kernel",
    head=None,
    sparse=None,
    out_dir=None,
    avr_gcc: str = "avr-gcc",
) -> AvrObjectBundle:
    """Compile `qmodel` to a Flash-resident AVR object via avr-gcc.

    `qmodel` is an `AffineQuantizedModel`. `mcu`/`f_cpu`/`opt` are passed to
    avr-gcc (`-mmcu`, `-DF_CPU`, the optimization flag). `head` is the readout
    head; `sparse="csr"` selects the CSR-sparse W_res kernel. When `out_dir` is
    given the bundle is also written there. Returns an `AvrObjectBundle`.

    Raises `RuntimeError` if `avr_gcc` is not on PATH.
    """
    from rclite.targets.arduino.emit_c import emit_affine_kernel_c

    if shutil.which(avr_gcc) is None:
        raise RuntimeError(
            f"export_avr_object needs {avr_gcc!r} on PATH (the AVR toolchain)"
        )
    info = info_from_affine(qmodel, name=name, head=head)
    src = emit_affine_kernel_c(qmodel, head=head, sparse=sparse)
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "kernel.c").write_text(src)
        obj = td / f"{name}.o"
        cmd = [
            avr_gcc,
            f"-mmcu={mcu}",
            opt,
            f"-DF_CPU={int(f_cpu)}UL",
            "-ffunction-sections",
            "-fdata-sections",
            "-c",
            str(td / "kernel.c"),
            "-o",
            str(obj),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"avr-gcc failed:\n{r.stderr[:2000]}")
        object_code = obj.read_bytes()

    header = emit_c_header(info)
    bundle = AvrObjectBundle(
        name=name,
        info=info,
        mcu=mcu,
        object_code=object_code,
        header=header,
        readme=_readme(name, info, mcu, opt),
    )
    if out_dir is not None:
        bundle.write(out_dir)
    return bundle
