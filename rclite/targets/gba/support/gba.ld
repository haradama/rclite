/* Linker script for the Game Boy Advance (ARM7TDMI).
 *   32 MB cartridge ROM @ 0x08000000  (code + rodata + .data init image)
 *   256 KB EWRAM        @ 0x02000000  (.data / .bss)
 *    32 KB IWRAM        @ 0x03000000  (stack)
 */
ENTRY(_start)

MEMORY
{
    ROM   (rx)  : ORIGIN = 0x08000000, LENGTH = 32M
    IWRAM (rwx) : ORIGIN = 0x03000000, LENGTH = 32K
    EWRAM (rwx) : ORIGIN = 0x02000000, LENGTH = 256K
}

SECTIONS
{
    .text : {
        KEEP(*(.crt0))          /* entry + header must be first in ROM */
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > ROM

    _sidata = LOADADDR(.data);
    .data : {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > EWRAM AT > ROM

    .bss : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > EWRAM

    /* libgcc's unwind code references these even though we never unwind. */
    .ARM.exidx : {
        __exidx_start = .;
        *(.ARM.exidx*)
        __exidx_end = .;
    } > ROM

    /* GBA BIOS reserves the top 0x100 bytes of IWRAM; mirror that here. */
    _stack_top = ORIGIN(IWRAM) + LENGTH(IWRAM) - 0x100;

    /DISCARD/ : { *(.ARM.extab*) }
}
