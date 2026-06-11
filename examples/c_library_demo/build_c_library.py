"""Build a shared library from a trained ESN and use it from a tiny C program.

Pipeline:
    1. Train an ESN on Mackey-Glass (Python).
    2. JIT-compile via LLVM → emit a PIC object → link with gcc to librc.so.
    3. Generate a C header declaring `rc_predict`.
    4. Write a minimal sample.c that calls the library.
    5. Compile sample.c against librc.so and run it.
    6. Cross-check the C output against the Python reference.

Artifacts land in ./build/ relative to this file.
"""

from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys
import textwrap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.targets import HostTarget

from examples.forecasting.mackey_glass_esn import mackey_glass


HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE.parents[1] / "build"


def train_esn() -> tuple[ReservoirComputer, RCExecutor, np.ndarray]:
    series = mackey_glass(n=2500)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    input_offset = float(X[:n_train].mean())

    rc = ReservoirComputer(
        input=InputNode(
            units=1, input_scaling=1.0, input_offset=input_offset, name="in"
        ),
        reservoir=ReservoirNode(
            units=200,
            activation=Activation.TANH,
            topology=Topology.SCR,
            chain_weight=0.9,
            leak_rate=0.3,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=200,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])
    return rc, exe, X[n_train : n_train + 10]


def main() -> None:
    if shutil.which("gcc") is None:
        sys.exit("error: gcc is required on PATH")
    BUILD.mkdir(parents=True, exist_ok=True)

    rc, exe, sample_X = train_esn()

    target = HostTarget()
    artifact = target.compile(rc, exe, output_dir=BUILD, lib_name="rc")
    lib_path = artifact.binary
    hdr_path = next(s for s in artifact.sources if s.suffix == ".h")
    src_path = BUILD / "rc_demo.c"
    exe_path = BUILD / "rc_demo"

    Y_ref = artifact.metadata["jit"].predict(sample_X)

    print(
        f"[1/4] emit shared library      -> {lib_path.relative_to(HERE.parents[1])}"
    )
    print(
        f"[2/4] emit C header            -> {hdr_path.relative_to(HERE.parents[1])}"
    )

    print(
        f"[3/4] write sample C program   -> {src_path.relative_to(HERE.parents[1])}"
    )
    flat = sample_X.ravel()
    x_literals = ",\n        ".join(f"{v:.17g}" for v in flat)
    src_path.write_text(
        textwrap.dedent(f"""\
        /* Minimal demo for the compiled ReservoirComputer.
         * Build:
         *   gcc -O2 -o rc_demo rc_demo.c -L. -lrc -Wl,-rpath,'$ORIGIN' -lm
         */
        #include <stdio.h>
        #include <stdint.h>
        #include "rc_predict.h"

        #define T 10

        int main(void) {{
            double X[T] = {{
                {x_literals}
            }};
            double Y[T] = {{0}};

            rc_predict((int64_t)T, X, Y);

            puts("  t       x            y");
            puts("  --------------------------------");
            for (int t = 0; t < T; t++) {{
                printf("  %2d  %10.5f  %12.6f\\n", t, X[t], Y[t]);
            }}
            return 0;
        }}
        """)
    )

    print(
        f"[4/4] gcc compile + link       -> {exe_path.relative_to(HERE.parents[1])}"
    )
    cmd = [
        "gcc",
        "-O2",
        "-Wall",
        "-o",
        str(exe_path),
        str(src_path),
        "-L",
        str(BUILD),
        "-lrc",
        f"-Wl,-rpath,{BUILD}",
        "-lm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"compile failed: {result.stderr}")

    print(f"\n--- Python reference output ---")
    for t in range(len(sample_X)):
        print(f"   t={t:2d}  x={sample_X[t, 0]:10.5f}  y={Y_ref[t, 0]:12.6f}")

    print(
        f"\n--- C demo output (./{exe_path.relative_to(HERE.parents[1])}) ---"
    )
    out = subprocess.run(
        [str(exe_path)], capture_output=True, text=True, check=True
    )
    print(out.stdout)


if __name__ == "__main__":
    main()
