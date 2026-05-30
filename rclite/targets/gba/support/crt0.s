@ Minimal Game Boy Advance (ARM7TDMI / ARMv4T) startup.
@
@   - The cartridge is mapped at 0x08000000. On (direct) boot the CPU starts
@     in ARM state with PC = 0x08000000, so the first word must be an ARM
@     branch over the 192-byte cartridge header to _start.
@   - mGBA's HLE BIOS direct-boot does not require a valid Nintendo logo, so
@     the header area is reserved as zeros.
@   - _start sets the IRQ and System mode stacks, installs a VBlank interrupt
@     handler, enables interrupts, copies .data, zeroes .bss, then calls main()
@     with ARMv4T interworking (bx).
@
@   Enabling and servicing the VBlank IRQ matters: without periodic interrupt
@   activity, mGBA resets the emulated console partway through a long compute
@   burst (e.g. a multi-step rc_predict). A real GBA program (and the gba-crate
@   runtime) always runs with interrupts enabled, so we mirror that here.
    .section .crt0, "ax"
    .arm
    .align 2
    .global _start

_rom_header:
    b       _start              @ 0x08000000: branch over the cartridge header
    .space  188, 0              @ 192-byte header (logo/title/...) — zeros are OK on mGBA

_start:
    @ IRQ mode (0x12), I+F masked while we set things up; set the IRQ stack.
    mov     r0, #0x92
    msr     cpsr_cf, r0
    ldr     sp, =0x03007FA0

    @ System mode (0x1f) with IRQ enabled (I=0), FIQ masked (F=1) => 0x5f.
    mov     r0, #0x5f
    msr     cpsr_cf, r0
    ldr     sp, =_stack_top

    @ Install the user IRQ handler pointer the BIOS dispatches through.
    ldr     r0, =irq_handler
    ldr     r1, =0x03007FFC
    str     r0, [r1]

    @ Fast GamePak ROM access + prefetch (WAITCNT) — the default boot value
    @ runs code from ROM at the slowest wait states; this is ~1.8x faster.
    ldr     r0, =0x04000204
    ldr     r1, =0x4317
    strh    r1, [r0]            @ WAITCNT

    @ Enable VBlank IRQ: DISPSTAT.bit3, IE=VBlank, IME=1.
    ldr     r0, =0x04000004
    mov     r1, #0x08
    strh    r1, [r0]            @ DISPSTAT: VBlank IRQ enable
    ldr     r0, =0x04000200
    mov     r1, #0x01
    strh    r1, [r0]            @ IE: VBlank
    ldr     r0, =0x04000208
    strh    r1, [r0]            @ IME: master enable

    @ Copy .data : ROM -> RAM.
    ldr     r0, =_sidata
    ldr     r1, =_sdata
    ldr     r2, =_edata
1:  cmp     r1, r2
    ldrlo   r3, [r0], #4
    strlo   r3, [r1], #4
    blo     1b

    @ Zero .bss.
    ldr     r1, =_sbss
    ldr     r2, =_ebss
    mov     r3, #0
2:  cmp     r1, r2
    strlo   r3, [r1], #4
    blo     2b

    @ Call main() (Thumb) — ARMv4T has no blx, so interwork via bx.
    ldr     r0, =main
    mov     lr, pc
    bx      r0
3:  b       3b                  @ spin if main ever returns

@ ARM-mode VBlank IRQ handler. The BIOS enters here in IRQ mode; we just
@ acknowledge the interrupt (hardware REG_IF and the BIOS IF mirror) and return.
    .arm
    .align 2
irq_handler:
    ldr     r0, =0x04000202     @ REG_IF
    ldrh    r1, [r0]
    strh    r1, [r0]            @ acknowledge hardware IF
    ldr     r0, =0x03007FF8     @ BIOS IF mirror at IWRAM top
    ldrh    r2, [r0]
    orr     r2, r2, r1
    strh    r2, [r0]
    bx      lr

    .section .note.GNU-stack, "", %progbits
