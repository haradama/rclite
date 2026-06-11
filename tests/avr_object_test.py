"""export_avr_object: Flash-resident, bit-exact AVR kernel object.

Builds an i8 affine ESN, exports it as an avr-gcc `.o` via `export_avr_object`,
links the object into a tiny firmware, and runs it on the emulated ATmega328P
(simavr). Verifies:

  * the object is an AVR ELF (EM_AVR),
  * the linked firmware is Flash-resident — weights live in PROGMEM/Flash, so
    SRAM usage stays tiny (far below the weight-table size),
  * the firmware's UART output is byte-for-byte identical to the Python
    `AffineQuantizedExecutor` (bit-exact on real AVR semantics).

Skips when the AVR toolchain (avr-gcc + simavr) is not on PATH.
"""

from __future__ import annotations

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
from rclite.export import export_avr_object, AvrObjectBundle

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
HAVE = (
    shutil.which("avr-gcc") is not None and shutil.which("simavr") is not None
)


def _model(K=3, M=2, N=32, T=6, seed=5):
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
    X = rng.standard_normal((200, K)) * 0.3
    Y = np.stack(
        [np.sin(np.arange(200) * 0.05 * (k + 1)) for k in range(M)], axis=1
    )
    exe.fit(X, Y)
    qm = quantize_model_affine(
        rc, exe, calibrate_from_data(rc, exe, X, storage_bits=8)
    )
    return qm, X[100 : 100 + T], T, K, M, N


def _executor_ref(qm, Xt, T):
    qe = AffineQuantizedExecutor(qm)
    qe.reset()
    ref = []
    for t in range(T):
        xr = qe._quantize_raw_input(Xt[t])
        qe.step_q(qe._quantize_u_pre(Xt[t]))
        ref += [int(v) for v in qe.predict_one_q(xr, qe.state_q)]
    return ref


def _firmware_c(Xq, T, K, M):
    xcsv = ",".join(str(int(v)) for v in Xq.reshape(-1))
    return f"""
#include <avr/io.h>
#include <stdint.h>
#include "rc_kernel.h"
static const int8_t X[{T * K}] = {{{xcsv}}};
static int8_t Yb[{T * M}];
static void tx(char c){{ while(!(UCSR0A&(1<<UDRE0))); UDR0=c; }}
static void pi(int v){{ char b[8]; int i=0; if(v<0){{tx('-');v=-v;}}
  if(!v){{tx('0');return;}} while(v){{b[i++]='0'+v%10;v/=10;}} while(i)tx(b[--i]); }}
int main(void){{
  UBRR0L=103; UCSR0B=(1<<TXEN0); UCSR0C=(1<<UCSZ01)|(1<<UCSZ00);
  rc_predict({T}, X, Yb);
  for(int i=0;i<{T * M};i++){{ pi((int)Yb[i]); tx(' '); }}
  tx(10); tx('E'); tx('N'); tx('D'); tx(10);
  for(;;);
}}
"""


def _run_simavr(elf, mcu, timeout=30):
    """Run `elf` on simavr; return the UART text (simavr emits it on stderr)."""
    p = subprocess.Popen(
        ["simavr", "-m", mcu, str(elf)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    import time

    buf = ""
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            line = p.stderr.readline()
            if not line:
                break
            buf += line
            if "END" in buf:
                break
    finally:
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
    return re.sub(r"\x1b\[[0-9;]*m", "", buf)  # strip ANSI color


def test_export_avr_object_flash_and_bitexact():
    if not HAVE:
        print("  (skip: avr-gcc / simavr not on PATH)")
        return
    mcu = "atmega328p"
    qm, Xt, T, K, M, N = _model()
    ref = _executor_ref(qm, Xt, T)
    Xq = np.ascontiguousarray(
        qm.config.input.quantize_array(Xt), dtype=np.int8
    )

    td = pathlib.Path(tempfile.mkdtemp())
    bundle = export_avr_object(qm, mcu=mcu, name="rc_kernel", out_dir=td)
    assert isinstance(bundle, AvrObjectBundle)
    for fn in ("rc_kernel.o", "rc_kernel.h", "README.md"):
        assert (td / fn).exists(), f"missing {fn}"

    # the object is an AVR ELF (e_machine == EM_AVR == 0x53, little-endian half)
    assert bundle.object_code[:4] == b"\x7fELF"
    assert bundle.object_code[18:20] == b"\x53\x00", "object is not EM_AVR"

    # link a firmware that drives the kernel and prints over UART
    (td / "fw.c").write_text(_firmware_c(Xq, T, K, M))
    elf = td / "fw.elf"
    r = subprocess.run(
        [
            "avr-gcc",
            f"-mmcu={mcu}",
            "-Os",
            "-DF_CPU=16000000UL",
            "-I",
            str(td),
            str(td / "fw.c"),
            str(td / "rc_kernel.o"),
            "-o",
            str(elf),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"link failed:\n{r.stderr}"

    # Flash-residency: SRAM (Data) must be far below the weight-table size.
    size = subprocess.run(
        ["avr-size", "-C", f"--mcu={mcu}", str(elf)],
        capture_output=True,
        text=True,
    ).stdout
    data = int(re.search(r"Data:\s+(\d+)\s+bytes", size).group(1))
    weight_bytes = N * N  # W_res alone; in Flash it must NOT sit in SRAM
    assert data < weight_bytes // 2, (
        f"SRAM {data}B too large vs weights {weight_bytes}B — not Flash-resident"
    )

    # bit-exact on the emulated ATmega328P
    out = _run_simavr(elf, mcu)
    got = [int(x) for x in re.findall(r"-?\d+", out.split("END")[0])]
    assert got == ref, f"AVR vs executor differ:\n got {got}\n ref {ref}"
    print(
        f"  export_avr_object: N={N} i8 kernel, Flash-resident (SRAM {data}B "
        f"<< weights {weight_bytes}B), bit-exact on simavr ({len(ref)} outputs)"
    )


def test_export_avr_object_requires_toolchain(monkeypatch=None):
    """Without out_dir it still returns a bundle; missing avr-gcc raises."""
    if not HAVE:
        print("  (skip)")
        return
    qm, Xt, T, K, M, N = _model(N=16, T=2)
    b = export_avr_object(qm, mcu="atmega328p")
    assert b.object_code[:4] == b"\x7fELF" and b.header.find("rc_predict") >= 0
    # a bogus compiler name raises a clear error
    try:
        export_avr_object(qm, avr_gcc="avr-gcc-does-not-exist")
        raise AssertionError("expected RuntimeError for missing avr-gcc")
    except RuntimeError as e:
        assert "PATH" in str(e)
    print(
        "  export_avr_object: returns bundle bytes; missing toolchain errors"
    )


TESTS = [
    test_export_avr_object_flash_and_bitexact,
    test_export_avr_object_requires_toolchain,
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
