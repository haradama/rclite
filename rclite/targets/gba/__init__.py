"""Game Boy Advance (ARM7TDMI / ARMv4T) deployment target."""

from .target import GbaTarget


class Gba(GbaTarget):
    """Convenience preset: GBA cartridge runnable under mGBA."""

    def __init__(self, dtype: str = "f32"):
        super().__init__(dtype=dtype)


__all__ = ["GbaTarget", "Gba"]
