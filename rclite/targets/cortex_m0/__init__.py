"""Cortex-M0(+) deployment target."""
from .target import CortexM0Target
from .boards import CortexM0Board, MicrobitV1, RaspberryPiPico


class Microbit(CortexM0Target):
    """Convenience preset: BBC micro:bit v1 (nRF51822, Cortex-M0) on QEMU."""
    def __init__(self, dtype: str = "f32"):
        super().__init__(board=MicrobitV1(), dtype=dtype)


class Pico(CortexM0Target):
    """Convenience preset: Raspberry Pi Pico (RP2040, Cortex-M0+).

    Produces a flash-resident ELF (boot_stage2 at 0x10000000, vector table
    at 0x10000100, .data RAM-resident with LMA in flash). The ELF is valid
    for both:
      - SWD loading via picoprobe + openocd (semihosting through openocd)
      - Wokwi RP2040 simulator (boots correctly through boot2)

    Caveat: Wokwi's Pi Pico simulator captures USB-CDC or wired-UART
    output, not ARM semihosting. The rc_predict template uses semihosting,
    so the ELF boots but produces no visible output in wokwi-cli yet —
    add UART init + a serial-monitor in diagram.json to surface output.

    No QEMU machine exists for RP2040.
    """
    def __init__(self, dtype: str = "f32"):
        super().__init__(board=RaspberryPiPico(), dtype=dtype)


__all__ = [
    "CortexM0Target", "CortexM0Board",
    "MicrobitV1", "Microbit",
    "RaspberryPiPico", "Pico",
]
