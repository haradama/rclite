"""Raspberry Pi Pico (RP2040) board descriptor."""
from __future__ import annotations
from dataclasses import dataclass

from .microbit_v1 import CortexM0Board


@dataclass(frozen=True)
class RaspberryPiPico(CortexM0Board):
    """Raspberry Pi Pico (RP2040: dual Cortex-M0+, 264 KB SRAM, 2 MB QSPI flash).

    Build target produces a RAM-resident ELF suitable for SWD loading via
    picoprobe + openocd (`monitor reset init; load; monitor arm semihosting
    enable; continue`). Flash-resident images would additionally require a
    second-stage bootloader (boot2) and are out of scope here.

    QEMU has no `-machine` for the RP2040, so `qemu_machine` is empty and
    `CortexM0Target.run()` will refuse with a clear message.
    """
    name: str = "raspberry-pi-pico"
    soc: str = "RP2040"
    flash_kb: int = 2048
    ram_kb: int = 264
    qemu_machine: str = ""
    linker_script: str = "rp2040.ld"
    cpu: str = "cortex-m0plus"
    wokwi_part: str = "wokwi-pi-pico"
    extra_asm: tuple = ("boot2_rp2040.S", "flash_entry_rp2040.S")
    # UART-based template lets Wokwi's $serialMonitor capture output;
    # semihosting (the default template) is not visible in Wokwi.
    main_template: str = "main_template_pico.c"
    main_template_q: str = "main_template_pico_q.c"
