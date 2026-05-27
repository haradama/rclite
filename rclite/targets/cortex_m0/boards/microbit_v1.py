"""Board descriptors for Cortex-M0 targets."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CortexM0Board:
    """Describes a Cortex-M0(+) board: SoC, memory layout, linker, runners.

    `qemu_machine` empty string means no QEMU model is available.
    `wokwi_part` empty string means no Wokwi simulator part is available.
    `extra_asm` is a tuple of .S filenames (in support/) to assemble alongside
    `startup.c` — used for board-specific boot stubs (e.g. RP2040 boot2).
    `CortexM0Target.run()` tries QEMU first, then Wokwi, then errors with
    on-device (SWD) instructions if neither is wired up.
    `cpu` is the LLVM/GCC `-mcpu=` value (cortex-m0 vs cortex-m0plus).
    """
    name: str
    soc: str
    flash_kb: int
    ram_kb: int
    qemu_machine: str
    linker_script: str  # filename within rclite/targets/cortex_m0/support/
    cpu: str = "cortex-m0"
    wokwi_part: str = ""
    extra_asm: tuple = ()
    # `main_template` and `main_template_q` are the C source templates the
    # board uses for f32 and i32 builds respectively. Boards that need a
    # different output channel (e.g. UART on Pico for Wokwi capture) point
    # to their own variants in support/.
    main_template: str = "main_template.c"
    main_template_q: str = "main_template_q.c"


@dataclass(frozen=True)
class MicrobitV1(CortexM0Board):
    """BBC micro:bit v1 (nRF51822: 256 KB flash, 16 KB SRAM, Cortex-M0)."""
    name: str = "microbit-v1"
    soc: str = "nRF51822"
    flash_kb: int = 256
    ram_kb: int = 16
    qemu_machine: str = "microbit"
    linker_script: str = "nrf51.ld"
    cpu: str = "cortex-m0"
    wokwi_part: str = ""
    extra_asm: tuple = ()
    main_template: str = "main_template.c"
    main_template_q: str = "main_template_q.c"
