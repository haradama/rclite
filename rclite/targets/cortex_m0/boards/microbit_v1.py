"""Board descriptors for Cortex-M0 targets."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CortexM0Board:
    """Describes a Cortex-M0 board: SoC, memory layout, linker, QEMU machine."""

    name: str
    soc: str
    flash_kb: int
    ram_kb: int
    qemu_machine: str
    linker_script: str  # filename within rclite/targets/cortex_m0/support/


@dataclass(frozen=True)
class MicrobitV1(CortexM0Board):
    """BBC micro:bit v1 (nRF51822: 256 KB flash, 16 KB SRAM, Cortex-M0)."""

    name: str = "microbit-v1"
    soc: str = "nRF51822"
    flash_kb: int = 256
    ram_kb: int = 16
    qemu_machine: str = "microbit"
    linker_script: str = "nrf51.ld"
