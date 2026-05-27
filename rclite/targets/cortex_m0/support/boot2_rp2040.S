/* Minimal RP2040 boot_stage2 stub.
 *
 * RP2040 bootrom requires exactly 256 bytes at flash start: 252 bytes of
 * code/data followed by a 4-byte CRC32 (MPEG-2 polynomial). The bootrom
 * verifies the CRC, then calls boot2 with `bl`. Our stub just returns;
 * the bootrom then branches to flash + 0x100 (the __flash_entry stub).
 *
 * Wokwi's RP2040 simulator handles XIP automatically, so we skip the
 * QSPI fast-read setup that the Pico SDK's full boot2 performs. This
 * means the binary boots on Wokwi but is NOT suitable for real-hardware
 * flashing (where boot2 must configure the QSPI for XIP reads).
 *
 * The 4-byte CRC at the end is pre-computed for the specific byte
 * sequence `0x70 0x47` + 250 zeros, using MPEG-2 CRC32
 * (poly 0x04C11DB7, init 0xFFFFFFFF, no reflect, no xor): 0xbfb5e005.
 */
    .syntax unified
    .cpu    cortex-m0plus
    .thumb

    .section .boot2, "ax"
    .global  __boot2
    .type    __boot2, %function
    .thumb_func
__boot2:
    bx      lr                          /* 2 bytes: return to bootrom */
    .fill   250, 1, 0x00                /* pad to 252 bytes */
    .byte   0x05, 0xe0, 0xb5, 0xbf      /* CRC32 (MPEG-2), little-endian */
    .size   __boot2, . - __boot2
