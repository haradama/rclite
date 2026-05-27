/* RP2040 entry stub at flash + 0x100.
 *
 * After boot_stage2 returns, the RP2040 bootrom branches to flash+0x100
 * as code (not as a vector table). We need a few instructions to load
 * SP and PC from the actual vector table, which sits immediately after.
 *
 * The bootrom does NOT auto-load SP from vector_table[0]: that's only
 * done at hardware reset for the bootrom's own vector table. So we do
 * it manually here before branching into Reset_Handler.
 */
    .syntax unified
    .cpu    cortex-m0plus
    .thumb

    .section .flash_entry, "ax"
    .global  __flash_entry
    .type    __flash_entry, %function
    .thumb_func
__flash_entry:
    ldr     r0, =vector_table
    ldr     r1, [r0]                /* vector_table[0] = initial SP */
    msr     msp, r1
    ldr     r0, [r0, #4]            /* vector_table[1] = Reset_Handler (thumb) */
    bx      r0
    .size   __flash_entry, . - __flash_entry
