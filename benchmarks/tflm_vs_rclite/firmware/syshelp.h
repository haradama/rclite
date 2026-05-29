/* Shared bare-metal helpers for the micro:bit (nRF51, Cortex-M0) firmwares:
 *   - ARM semihosting stdout (BKPT #0xAB) + clean exit
 *   - integer decimal/hex formatting (no libc printf, no soft-FP)
 *   - SysTick read for a deterministic instruction-proportional "cycle" count
 *     under qemu -icount (NOT cycle-accurate; valid only as a like-for-like
 *     relative measure between the two firmwares built the same way).
 *
 * C and C++ compatible so the rclite (C) and TFLM (C++) firmwares share it.
 */
#ifndef SYSHELP_H_
#define SYSHELP_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline void sh_puts(const char *s) {
    register int r0 __asm__("r0") = 0x04;        /* SYS_WRITE0 */
    register const char *r1 __asm__("r1") = s;
    __asm__ volatile("bkpt #0xab" : "+r"(r0) : "r"(r1) : "memory");
}

__attribute__((noreturn)) static inline void sh_exit(int code) {
    register int r0 __asm__("r0") = 0x18;         /* SYS_EXIT */
    register int r1 __asm__("r1") = (code == 0) ? 0x20026 : code;
    __asm__ volatile("bkpt #0xab" : "+r"(r0) : "r"(r1) : "memory");
    while (1) {
    }
}

static inline int sh_fmt_int(char *buf, int32_t v) {
    char *p = buf;
    if (v < 0) { *p++ = '-'; v = -v; }
    char tmp[12];
    int n = 0;
    if (v == 0) tmp[n++] = '0';
    else while (v > 0) { tmp[n++] = (char)('0' + v % 10); v /= 10; }
    while (n > 0) *p++ = tmp[--n];
    *p = 0;
    return (int)(p - buf);
}

static inline void sh_put_kv(const char *k, int32_t v) {
    char b[16];
    sh_puts(k);
    sh_fmt_int(b, v);
    sh_puts(b);
    sh_puts("\n");
}

/* ---- deterministic instruction-proportional clock --------------------
 * qemu does NOT model Cortex-M pipeline/cycle timing, so SysTick is useless
 * here. Under `qemu -icount shift=0` the semihosting SYS_ELAPSED tick count
 * advances one per executed (virtual) instruction, giving a deterministic,
 * reproducible op-count proxy. It is NOT real silicon cycles, but it is a
 * valid like-for-like comparison between the two firmwares run identically.
 */
static inline uint64_t sh_elapsed(void) {
    uint32_t buf[2] = {0, 0};
    register int r0 __asm__("r0") = 0x30;          /* SYS_ELAPSED */
    register uint32_t *r1 __asm__("r1") = buf;
    __asm__ volatile("bkpt #0xab" : "+r"(r0) : "r"(r1) : "memory");
    return ((uint64_t)buf[1] << 32) | buf[0];
}

#ifdef __cplusplus
}
#endif
#endif  /* SYSHELP_H_ */
