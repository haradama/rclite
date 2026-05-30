"""Uniform metadata for a deployable quantized kernel.

`KernelInfo` captures everything the portable C header and the Rust
wrapper need, independent of which quantization family produced the
model. Both the asymmetric *affine* path and the symmetric *Q-format*
path collapse to the same `(scale, zero_point)` description of the input
and output tensors:

    quantize(x)   = round(x / in_scale)  + in_zp        (saturated)
    dequantize(q) = (q - out_zp) * out_scale

For the symmetric path `in_zp = out_zp = 0` and the scales are powers of
two (`2^-input_frac`, `2^-state_frac`).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelInfo:
    """Target-agnostic description of an emitted `rc_predict` kernel."""

    name: str               # short model name (used for the Rust crate)
    quant: str              # "affine" | "symmetric"
    storage_bits: int       # 8 / 16 (/ 32 for symmetric)
    K: int                  # input dim
    M: int                  # output dim
    N: int                  # reservoir units
    topology: str
    # Input quantization  (float x -> q):  q = round(x/in_scale) + in_zp
    in_scale: float
    in_zp: int
    # Output dequantization (q -> float y): y = (q - out_zp) * out_scale
    out_scale: float
    out_zp: int
    # Classification head: "logits" (regression / raw scores), "classify"
    # (argmax class id, one int32 per step), or "proba" (M probabilities at
    # Q.prob_frac per step). `task`/`n_classes` describe the readout.
    head: str = "logits"
    task: str = "REGRESSION"
    n_classes: int = 0

    @property
    def is_classify(self) -> bool:
        return self.head == "classify"

    @property
    def is_proba(self) -> bool:
        return self.head == "proba"

    @property
    def prob_frac(self) -> int:
        return min(self.storage_bits - 1, 15)

    @property
    def out_ctype(self) -> str:
        return "int32_t" if self.is_classify else self.storage_ctype

    @property
    def out_rust(self) -> str:
        return "i32" if self.is_classify else self.storage_rust

    @property
    def storage_ctype(self) -> str:
        return {8: "int8_t", 16: "int16_t", 32: "int32_t"}[self.storage_bits]

    @property
    def storage_rust(self) -> str:
        return {8: "i8", 16: "i16", 32: "i32"}[self.storage_bits]

    @property
    def qmin(self) -> int:
        return -(1 << (self.storage_bits - 1))

    @property
    def qmax(self) -> int:
        return (1 << (self.storage_bits - 1)) - 1


def _head_meta(qmodel):
    ro = qmodel.rc.readout
    task = ro.task.name
    n_classes = ro.units if task == "CLASSIFICATION" else 0
    return task, n_classes


def info_from_affine(qmodel, name: str = "rc_model", *, head=None) -> KernelInfo:
    cfg = qmodel.config
    task, n_classes = _head_meta(qmodel)
    return KernelInfo(
        name=name, quant="affine",
        storage_bits=qmodel.storage_bits,
        K=qmodel.K, M=qmodel.M, N=qmodel.N,
        topology=qmodel.rc.reservoir.topology.name,
        in_scale=float(cfg.input.scale), in_zp=int(cfg.input.zero_point),
        out_scale=float(cfg.output.scale), out_zp=int(cfg.output.zero_point),
        head=head or "logits", task=task, n_classes=n_classes,
    )


def info_from_symmetric(qmodel, name: str = "rc_model", *, head=None) -> KernelInfo:
    cfg = qmodel.config
    task, n_classes = _head_meta(qmodel)
    return KernelInfo(
        name=name, quant="symmetric",
        storage_bits=qmodel.target.storage_bits,
        K=qmodel.K, M=qmodel.M, N=qmodel.N,
        topology=qmodel.rc.reservoir.topology.name,
        # symmetric input is quantized at 2^input_frac with zero_point 0
        in_scale=1.0 / float(1 << cfg.input_frac), in_zp=0,
        # output is reported at state scale = 2^state_frac
        out_scale=1.0 / float(1 << cfg.state_frac), out_zp=0,
        head=head or "logits", task=task, n_classes=n_classes,
    )
