/* Linker script for RP2040 (Raspberry Pi Pico), flash-resident.
 *   2 MB XIP flash @ 0x10000000   (boot2 + .text + .rodata + .data initials)
 *   264 KB SRAM    @ 0x20000000   (.data + .bss + stack)
 *
 *   0x10000000–0x100000FF : boot_stage2 (256 B, CRC32 in last 4 B)
 *   0x10000100            : __flash_entry stub
 *                           vector_table immediately after
 *                           .text, .rodata follow
 *
 * Boot flow (RP2040 bootrom):
 *   1. Read first 256 B of flash; verify CRC32 (MPEG-2)
 *   2. `bl` to boot_stage2 (does QSPI setup; our stub just returns)
 *   3. Branch to flash + 0x100 → __flash_entry → Reset_Handler → main()
 *
 * Suitable for Wokwi simulation; real-hardware flashing requires a
 * full boot_stage2 (XIP fast-read setup) — see boot2_rp2040.S notes.
 */
ENTRY(__flash_entry)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x10000000, LENGTH = 2048K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 264K
}

SECTIONS
{
    .boot2 0x10000000 : {
        KEEP(*(.boot2))
    } > FLASH

    .text 0x10000100 : {
        KEEP(*(.flash_entry))
        KEEP(*(.vector_table))
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    _sidata = LOADADDR(.data);
    .data : {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    _stack_top = ORIGIN(RAM) + LENGTH(RAM);

    /DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) }
}
