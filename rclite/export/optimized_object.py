"""Ship a *pre-optimized* kernel object plus a clean C header.

`export_bundle` ships portable **source** (a scalar C99 kernel) that the
caller's own compiler optimizes — so the result is only as good as that
compiler. This module ships the opposite contract: rclite compiles the kernel
through MLIR/LLVM *here* (vectorized when the target has a SIMD ISA), emits a
relocatable `.o`, and hands the C side only a declaration. The caller LINKS the
already-optimized object — its compiler never has to reproduce the SIMD.

    export_optimized_object(qmodel, target="x86_64-avx2", out_dir="build/")
      -> build/rc_kernel.o   LLVM-optimized kernel (vectorized matvec)
         build/rc_kernel.h   memref ABI decl + clean `rc_run(T, X, Y)` wrapper
         build/README.md     how to compile a caller and link the .o

`target` is a key in `SIMD_TARGETS` ("x86_64-avx2", "aarch64-neon",
"armv7-neon", "wasm32-simd128", "riscv64-rvv", "x86_64-sse2", "host"), a
`SimdTarget`, or any rclite `targets.Target` instance (its `triple`/`features`
are reused). The default `vlen` follows the target's SIMD width; pass `vlen=` to
override (`vlen=1` emits the scalar kernel). The vectorized integer matvec is
bit-exact with the scalar executor — see
`tests/mlir_affine_spike_test.py::test_vectorized_matvec_bit_exact`.

Affine (asymmetric) models only — the affine emitter is the one with a `vlen`
SIMD knob. Use `export_bundle` for the symmetric Q-format path.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, replace

from .info import KernelInfo, info_from_affine
from ..codegen import mlir_jit  # llvmlite-only at import; xdsl is lazy


@dataclass(frozen=True)
class SimdTarget:
    """A cross-compile target for the optimized-object route.

    `triple`/`cpu`/`features` go straight to `llc`; `vlen` is the SIMD width the
    affine emitter vectorizes the matvec reductions to (1 = scalar).
    """

    key: str
    triple: str
    cpu: str
    features: str
    vlen: int
    desc: str


# Curated SIMD targets. `vlen` is chosen so one vector spans the register at the
# i32 accumulator width: 256-bit AVX2 -> 8 i32 lanes; 128-bit NEON/SSE2/simd128
# -> 4. Any vlen stays bit-exact (the i64/i32 reduction is associative); a wider
# vlen than the register just gets split by LLVM. llc carries every default LLVM
# backend, so these cross-compile with no external toolchain (only the *linker*
# for the caller is target-specific).
SIMD_TARGETS: dict[str, SimdTarget] = {
    "x86_64-avx2": SimdTarget(
        "x86_64-avx2",
        "x86_64-unknown-linux-gnu",
        "",
        "+avx2",
        8,
        "x86-64 AVX2 (256-bit)",
    ),
    "x86_64-sse2": SimdTarget(
        "x86_64-sse2",
        "x86_64-unknown-linux-gnu",
        "",
        "+sse2",
        4,
        "x86-64 SSE2 (128-bit)",
    ),
    "aarch64-neon": SimdTarget(
        "aarch64-neon",
        "aarch64-unknown-linux-gnu",
        "",
        "+neon",
        4,
        "AArch64 NEON (128-bit)",
    ),
    "armv7-neon": SimdTarget(
        "armv7-neon",
        "armv7-unknown-linux-gnueabihf",
        "",
        "+neon",
        4,
        "ARMv7-A NEON (128-bit)",
    ),
    "wasm32-simd128": SimdTarget(
        "wasm32-simd128",
        "wasm32-unknown-unknown",
        "",
        "+simd128",
        4,
        "WebAssembly SIMD128 (128-bit)",
    ),
    "riscv64-rvv": SimdTarget(
        "riscv64-rvv",
        "riscv64-unknown-linux-gnu",
        "",
        "+v",
        8,
        "RISC-V Vector (RVV)",
    ),
}

# arch prefix -> (features, vlen) for the auto-detected host target.
_HOST_SIMD = {
    "x86_64": ("+avx2", 8),
    "amd64": ("+avx2", 8),
    "aarch64": ("+neon", 4),
    "arm64": ("+neon", 4),
}


def _host_target() -> SimdTarget:
    """The build host as a `SimdTarget` (default triple, SIMD by arch)."""
    from llvmlite import binding as llvm

    triple = llvm.get_default_triple()
    arch = triple.split("-", 1)[0]
    features, vlen = _HOST_SIMD.get(arch, ("", 1))
    return SimdTarget(
        "host", triple, "", features, vlen, f"build host ({triple})"
    )


def _from_rclite_target(target) -> SimdTarget:
    """Adapt an rclite `targets.Target` (has `.triple`/`.features`)."""
    triple = getattr(target, "triple", None)
    if not triple:
        raise TypeError(
            f"{type(target).__name__} has no `triple`; pass a SIMD_TARGETS key "
            "or a SimdTarget instead"
        )
    features = getattr(target, "features", "") or ""
    cpu = getattr(target, "cpu", "") or ""
    # A SIMD feature implies a 128-bit-ish width; otherwise scalar.
    vlen = (
        4 if any(f in features for f in ("simd", "neon", "sse", "avx")) else 1
    )
    key = getattr(target, "name", None) or triple
    return SimdTarget(key, triple, cpu, features, vlen, f"rclite target {key}")


def _resolve(target, vlen) -> SimdTarget:
    if target is None or target == "host":
        spec = _host_target()
    elif isinstance(target, SimdTarget):
        spec = target
    elif isinstance(target, str):
        if target not in SIMD_TARGETS:
            raise KeyError(
                f"unknown target {target!r}; choose from "
                f"{['host', *SIMD_TARGETS]} or pass a SimdTarget"
            )
        spec = SIMD_TARGETS[target]
    elif hasattr(target, "triple"):
        spec = _from_rclite_target(target)
    else:
        raise TypeError(f"cannot resolve target from {target!r}")
    if vlen is not None:
        spec = replace(spec, vlen=int(vlen))
    return spec


@dataclass
class OptimizedObjectBundle:
    """An LLVM-optimized kernel `.o` plus the C header that exposes it."""

    name: str
    info: KernelInfo
    target: SimdTarget
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


def _guard(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper() + "_H"


def _readme(name: str, info: KernelInfo, spec: SimdTarget) -> str:
    vec = (
        f"SIMD-vectorized (vlen={spec.vlen}, {spec.desc})"
        if spec.vlen > 1
        else "scalar"
    )
    return f"""# {name} — pre-optimized rclite kernel

`{name}.o` is the reservoir kernel compiled by rclite through MLIR/LLVM for
**{spec.desc}** (`{spec.triple}`). The integer matvec reductions are {vec};
the kernel is **bit-exact** with rclite's Python reference executor.

You link this object — your compiler does not re-derive the optimization.

## Files

| file | role |
|------|------|
| `{name}.o` | optimized relocatable object (triple `{spec.triple}`) |
| `{name}.h` | `rc_predict` declaration + clean `rc_run(T, X, Y)` wrapper |

## Shape

- input dim  `RC_K = {info.K}`  (`rc_in_t` = `{info.storage_ctype}`)
- output dim `RC_M = {info.M}`  (`rc_out_t` = `{info.out_ctype}`)
- reservoir  `N = {info.N}` ({info.topology}), head `{info.head}`

## Use

```c
#include "{name}.h"

// X: T*RC_K inputs (row-major), Y: T*RC_M outputs (row-major).
rc_run(T, X, Y);
```

Compile the caller at **any** `-O` level — the kernel's speed lives in `{name}.o`:

```sh
cc -c my_app.c -o my_app.o
cc my_app.o {name}.o -o my_app        # native link
```

For a cross target use that toolchain's linker (e.g. `aarch64-linux-gnu-gcc`,
`wasm-ld` for `wasm32-simd128`). The `.o` already targets `{spec.triple}`.
"""


def export_optimized_object(
    qmodel,
    *,
    target="host",
    vlen=None,
    head=None,
    name: str = "rc_kernel",
    out_dir=None,
) -> OptimizedObjectBundle:
    """Compile `qmodel` to an optimized `.o` + C header for `target`.

    `qmodel` is an `AffineQuantizedModel`. `target` selects the ISA (see module
    docstring); `vlen` overrides the SIMD width; `head` is the readout head
    (`None`/`"classify"`/`"proba"`). When `out_dir` is given the bundle is also
    written there. Returns an `OptimizedObjectBundle` (object bytes + header +
    README).
    """
    # xDSL is an optional dep; keep `rclite.export` importable without it.
    from ..codegen.mlir_affine_xdsl import emit_affine_mlir_xdsl

    spec = _resolve(target, vlen)
    info = info_from_affine(qmodel, name=name, head=head)
    mlir_text = emit_affine_mlir_xdsl(qmodel, head=head, vlen=spec.vlen)
    extra = ["--convert-vector-to-llvm"] if spec.vlen > 1 else []
    obj = mlir_jit.cross_compile_object(
        mlir_text,
        triple=spec.triple,
        cpu=spec.cpu,
        features=spec.features,
        extra_passes=extra,
        filetype="obj",
    )
    header = mlir_jit.emit_c_header(
        K=qmodel.K,
        M=qmodel.M,
        storage_bits=qmodel.storage_bits,
        classify=(head == "classify"),
        func_name="rc_predict",
        guard=_guard(name),
    )
    bundle = OptimizedObjectBundle(
        name=name,
        info=info,
        target=spec,
        object_code=obj,
        header=header,
        readme=_readme(name, info, spec),
    )
    if out_dir is not None:
        bundle.write(out_dir)
    return bundle
