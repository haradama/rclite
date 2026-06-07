"""Cortex-M0 deployment target."""

from .target import CortexM0Target
from .boards import CortexM0Board, MicrobitV1


class Microbit(CortexM0Target):
    """Convenience preset: BBC micro:bit v1 (nRF51822) on QEMU."""

    def __init__(self, dtype: str = "f32"):
        super().__init__(board=MicrobitV1(), dtype=dtype)


__all__ = ["CortexM0Target", "CortexM0Board", "MicrobitV1", "Microbit"]
