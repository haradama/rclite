"""Nintendo Entertainment System (MOS 6502 / NROM) target via llvm-mos."""
from .target import NesTarget


class Nes(NesTarget):
    """Convenience preset: NES NROM cartridge runnable under Mesen."""
    def __init__(self, cc: str = "mos-nes-nrom-clang"):
        super().__init__(cc=cc)


__all__ = ["NesTarget", "Nes"]
