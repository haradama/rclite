/* Minimal linker script for the Arm Corstone-300 (SSE-300, Cortex-M55) — the
 * same FVP subsystem ExecuTorch's executor_runner targets. Code in ITCM,
 * data/stack in DTCM. Bare-metal; I/O via semihosting (see syshelp.h). */
ENTRY(Reset_Handler)

MEMORY
{
    ITCM (rx) : ORIGIN = 0x10000000, LENGTH = 0x00080000   /* 512 KiB code  */
    DTCM (rw) : ORIGIN = 0x30000000, LENGTH = 0x00080000   /* 512 KiB data  */
}

SECTIONS
{
    .vectors : {
        KEEP(*(.vectors))
        KEEP(*(.vector_table))
    } > ITCM

    .text : {
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > ITCM

    _sidata = LOADADDR(.data);
    .data : {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > DTCM AT > ITCM

    .bss : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > DTCM

    _stack_top = ORIGIN(DTCM) + LENGTH(DTCM);

    /DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) }
}
