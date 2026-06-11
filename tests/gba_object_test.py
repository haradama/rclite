"""export_gba_object: bit-exact GBA kernel + the ARM/IWRAM speedup.

Exports an i8 affine ESN for the GBA both ways — `thumb-rom` (Thumb code in
cartridge ROM) and `arm-iwram` (ARM code routed into 0-waitstate IWRAM) — links
each into a tiny GBA ROM, and runs both on the cycle-accurate mGBA. Verifies:

  * both kernels are bit-exact with the Python `AffineQuantizedExecutor`,
  * the arm-iwram kernel actually lands in IWRAM (0x03xxxxxx),
  * arm-iwram is materially faster than thumb-rom (the GBA optimization — there
    is no SIMD on ARMv4T; the win is ARM-vs-Thumb + IWRAM placement).

Skips when the GBA toolchain (arm-none-eabi-gcc + mgba) is not on PATH.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant.affine import (
    calibrate_from_data,
    quantize_model_affine,
    AffineQuantizedExecutor,
)
from rclite.export import export_gba_object, GbaObjectBundle
from rclite.codegen import mlir_jit

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
_MGBA = shutil.which("mgba") or shutil.which("/usr/games/mgba")
_SUPPORT = pathlib.Path(__file__).resolve().parent.parent / (
    "rclite/targets/gba/support"
)
HAVE = (
    shutil.which("arm-none-eabi-gcc") is not None
    and _MGBA is not None
    and mlir_jit.tools_available()
)
try:
    import xdsl  # noqa: F401

    HAVE = HAVE and True
except ImportError:
    HAVE = False

# One linker script for both modes: the arm-iwram kernel's `.iwram.rc.*` lands
# in IWRAM; in thumb-rom mode that section is empty and the kernel stays in ROM.
_LD = """\
ENTRY(_start)
MEMORY
{
    ROM   (rx)  : ORIGIN = 0x08000000, LENGTH = 32M
    IWRAM (rwx) : ORIGIN = 0x03000000, LENGTH = 32K
    EWRAM (rwx) : ORIGIN = 0x02000000, LENGTH = 256K
}
SECTIONS
{
    .text : { KEEP(*(.crt0)) *(.text*) *(.rodata*) . = ALIGN(4); } > ROM
    .rc_iwram : {
        . = ALIGN(4); __rc_iwram_start = .;
        *(.iwram.rc.text .iwram.rc.rodata)
        . = ALIGN(4); __rc_iwram_end = .;
    } > IWRAM AT > ROM
    __rc_iwram_lma = LOADADDR(.rc_iwram);
    _sidata = LOADADDR(.data);
    .data : { . = ALIGN(4); _sdata = .; *(.data*) . = ALIGN(4); _edata = .; } > EWRAM AT > ROM
    .bss  : { . = ALIGN(4); _sbss = .; *(.bss*) *(COMMON) . = ALIGN(4); _ebss = .; } > EWRAM
    PROVIDE(end = .); PROVIDE(_end = .); PROVIDE(__end__ = .);
    .ARM.exidx : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > ROM
    .ARM.extab : { *(.ARM.extab*) } > ROM
    _stack_top = ORIGIN(IWRAM) + LENGTH(IWRAM) - 0x100;
}
"""

_HARNESS = """\
#include <stdint.h>
#include "data.h"
#include "rc_kernel.h"
#include "mgba_log.h"
int _close(int f){(void)f;return -1;}
int _lseek(int f,int p,int w){(void)f;(void)p;(void)w;return 0;}
int _read(int f,char*p,int n){(void)f;(void)p;(void)n;return 0;}
int _write(int f,const char*p,int n){(void)f;(void)p;return n;}
int _kill(int p,int s){(void)p;(void)s;return -1;}
int _getpid(void){return 1;}
void _exit(int c){(void)c;for(;;);}
void abort(void){for(;;);}
extern char __rc_iwram_lma, __rc_iwram_start, __rc_iwram_end;
#define TM0L (*(volatile uint16_t*)0x4000100)
#define TM0H (*(volatile uint16_t*)0x4000102)
#define TM1L (*(volatile uint16_t*)0x4000104)
#define TM1H (*(volatile uint16_t*)0x4000106)
#define IME  (*(volatile uint16_t*)0x4000208)
static signed char Yout[RC_T*RC_M];
int main(void){
  mgba_open();
  for(char *d=&__rc_iwram_start,*s=&__rc_iwram_lma; d<&__rc_iwram_end;) *d++=*s++;
  IME=0;
  TM0H=0;TM1H=0;TM0L=0;TM1L=0; TM1H=0x84; TM0H=0x80;
  rc_run(RC_T, Xin, Yout);
  uint16_t hi=TM1L,lo=TM0L,hi2=TM1L; if(hi!=hi2){hi=hi2;lo=TM0L;}
  uint32_t cyc=((uint32_t)hi<<16)|lo;
  IME=1;
  int32_t chk=0; for(int i=0;i<RC_T*RC_M;i++) chk+=Yout[i];
  char b[160]; int n=0; const char*p="CYC=";
  while(*p)b[n++]=*p++;
  {char t[12];int k=0;uint32_t v=cyc;if(!v)t[k++]='0';while(v){t[k++]='0'+v%10;v/=10;}while(k)b[n++]=t[--k];}
  p=" CHK=";while(*p)b[n++]=*p++;
  {int32_t v=chk;if(v<0){b[n++]='-';v=-v;}char t[12];int k=0;if(!v)t[k++]='0';while(v){t[k++]='0'+v%10;v/=10;}while(k)b[n++]=t[--k];}
  p=(chk==REF_CHK)?" TEST_PASS":" TEST_FAIL"; while(*p)b[n++]=*p++; b[n]=0;
  mgba_log(b);
  for(;;);
}
"""


def _model(K=3, M=2, N=64, T=48, seed=5):
    rc = ReservoirComputer(
        input=InputNode(units=K, name="in"),
        reservoir=ReservoirNode(
            units=N,
            topology=Topology.ESN_STANDARD,
            leak_rate=0.3,
            density=0.4,
            seed=seed,
            name="res",
        ),
        readout=ReadoutNode(
            units=M,
            trainer=Trainer.RIDGE,
            regularization=1e-3,
            washout=10,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((400, K)) * 0.3
    Y = np.stack(
        [np.sin(np.arange(400) * 0.04 * (k + 1)) for k in range(M)], axis=1
    )
    exe.fit(X, Y)
    qm = quantize_model_affine(
        rc, exe, calibrate_from_data(rc, exe, X, storage_bits=8)
    )
    return qm, X[200 : 200 + T], T, K, M, N


def _ref_chk(qm, Xt, T):
    qe = AffineQuantizedExecutor(qm)
    qe.reset()
    s = 0
    for t in range(T):
        xr = qe._quantize_raw_input(Xt[t])
        qe.step_q(qe._quantize_u_pre(Xt[t]))
        s += int(sum(int(v) for v in qe.predict_one_q(xr, qe.state_q)))
    return s


def _build_and_run(td, bundle, Xq, T, K, M, ref_chk, tag):
    """Link the bundle's .o into a GBA ROM, run mGBA, return (cycles, passed)."""
    d = td / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "rc_kernel.o").write_bytes(bundle.object_code)
    (d / "rc_kernel.h").write_text(bundle.header)
    (d / "data.h").write_text(
        f"#define RC_T {T}\n#define RC_K {K}\n#define RC_M {M}\n"
        f"#define REF_CHK {ref_chk}\n"
        f"static const signed char Xin[{T * K}]={{"
        + ",".join(str(int(v)) for v in Xq.reshape(-1))
        + "};\n"
    )
    (d / "main.c").write_text(_HARNESS)
    (d / "rc.ld").write_text(_LD)
    shutil.copy(_SUPPORT / "crt0.s", d / "crt0.s")
    shutil.copy(_SUPPORT / "mgba_log.h", d / "mgba_log.h")
    cc = "arm-none-eabi-gcc"
    subprocess.run(
        [
            cc,
            "-marm",
            "-mthumb-interwork",
            "-c",
            str(d / "crt0.s"),
            "-o",
            str(d / "crt0.o"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            cc,
            "-mthumb",
            "-mthumb-interwork",
            "-Os",
            "-ffunction-sections",
            "-fdata-sections",
            "-I",
            str(d),
            "-c",
            str(d / "main.c"),
            "-o",
            str(d / "main.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    elf = d / "rc.elf"
    r = subprocess.run(
        [
            cc,
            "-mcpu=arm7tdmi",
            "-mthumb",
            "-mthumb-interwork",
            "-nostartfiles",
            "-Wl,--gc-sections",
            "--specs=nosys.specs",
            "-T",
            str(d / "rc.ld"),
            str(d / "crt0.o"),
            str(d / "main.o"),
            str(d / "rc_kernel.o"),
            "-o",
            str(elf),
            "-lgcc",
            "-lc",
            "-lnosys",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"{tag} link failed:\n{r.stderr[-1500:]}"
    rom = d / "rc.gba"
    subprocess.run(
        ["arm-none-eabi-objcopy", "-O", "binary", str(elf), str(rom)],
        check=True,
        capture_output=True,
    )

    # where did rc_predict land?
    nm = subprocess.run(
        ["arm-none-eabi-nm", str(elf)], capture_output=True, text=True
    ).stdout
    addr = next(
        (
            ln.split()[0]
            for ln in nm.splitlines()
            if ln.endswith(" rc_predict")
        ),
        "",
    )

    # mGBA never exits (the GBA has no clean halt); let coreutils `timeout` stop
    # it after it has printed, and capture the line-buffered debug log on stdout.
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    cmd = ["timeout", "8"] + (
        ["stdbuf", "-oL"] if shutil.which("stdbuf") else []
    )
    cmd += [_MGBA, "-l", "15", str(rom)]
    p = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=30
    )
    line = next(
        (ln for ln in (p.stdout + p.stderr).splitlines() if "CYC=" in ln), ""
    )
    cyc = (
        int(re.search(r"CYC=(\d+)", line).group(1)) if "CYC=" in line else None
    )
    return cyc, ("TEST_PASS" in line), addr


def test_export_gba_object_iwram_speedup():
    if not HAVE:
        print(
            "  (skip: arm-none-eabi-gcc / mgba / MLIR toolchain not available)"
        )
        return
    qm, Xt, T, K, M, N = _model()
    ref = _ref_chk(qm, Xt, T)
    Xq = np.ascontiguousarray(
        qm.config.input.quantize_array(Xt), dtype=np.int8
    )
    td = pathlib.Path(tempfile.mkdtemp())

    thumb = export_gba_object(
        qm, mode="thumb-rom", name="rc_kernel", out_dir=td / "tb"
    )
    arm = export_gba_object(
        qm, mode="arm-iwram", name="rc_kernel", out_dir=td / "aw"
    )
    assert isinstance(arm, GbaObjectBundle)
    assert arm.linker_fragment is not None and thumb.linker_fragment is None
    for fn in (
        "rc_kernel.o",
        "rc_kernel.h",
        "rc_kernel.iwram.ld",
        "README.md",
    ):
        assert (td / "aw" / fn).exists(), f"missing {fn}"

    ct, pt, at = _build_and_run(td, thumb, Xq, T, K, M, ref, "thumb")
    ca, pa, aa = _build_and_run(td, arm, Xq, T, K, M, ref, "arm")

    assert pt, "thumb-rom kernel not bit-exact on mGBA"
    assert pa, "arm-iwram kernel not bit-exact on mGBA"
    assert aa.startswith("0300"), f"arm-iwram kernel not in IWRAM (@{aa})"
    assert at.startswith("0800"), f"thumb-rom kernel not in ROM (@{at})"
    assert ca and ct and ca < ct, (
        f"arm-iwram ({ca}) should beat thumb-rom ({ct})"
    )
    print(
        f"  export_gba_object: both bit-exact on mGBA; thumb-rom={ct} cyc, "
        f"arm-iwram={ca} cyc (@IWRAM 0x{aa}) -> {ct / ca:.1f}x faster"
    )


def test_export_gba_object_modes_and_validation():
    if not HAVE:
        print("  (skip)")
        return
    qm, Xt, T, K, M, N = _model(N=24, T=4)
    b = export_gba_object(qm, mode="thumb-rom")
    assert b.object_code[:4] == b"\x7fELF" and "rc_run" in b.header
    try:
        export_gba_object(qm, mode="nonsense")
        raise AssertionError("expected ValueError for bad mode")
    except ValueError as e:
        assert "mode" in str(e)
    print("  export_gba_object: modes + validation ok")


TESTS = [
    test_export_gba_object_iwram_speedup,
    test_export_gba_object_modes_and_validation,
]


def main():
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"{PASS} {t.__name__}")
        except Exception:
            failures += 1
            print(f"{FAIL} {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
