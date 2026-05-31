/* Cortex-M0 SysTick op-count harness for the LLVM-cross-compiled rc_predict
 * (float OR integer storage — the storage type, dims, and parity tolerance
 * all come from the generated rc_data.h).
 *
 * Links against `void rc_predict(int64_t T, storage*, storage*)`. Timing uses
 * SysTick under `qemu -icount shift=0`: the virtual clock is driven by the
 * executed-instruction count, so the down-counter delta is DETERMINISTIC
 * (bit-stable run to run). SysTick advances at the CPU-clock rate (~1 tick
 * per ~62 instructions on nRF51), so the absolute number is "SysTick ticks",
 * proportional to executed instructions — NOT silicon cycles. Speedup ratios
 * equal instruction-count ratios.
 *
 * Parity uses a double-precision |diff| against RC_FW_EPS: 0.5 for integer
 * storage (exact → diff rounds to 0), a small tolerance for f32 (host f64
 * reference vs device f32 differ by rounding).
 *
 * rc_data.h provides rc_fw_storage_t, RC_FW_EPS, RC_FW_T/K/M, g_x[], g_y_ref[].
 */
#include <stdint.h>
#include "rc_data.h"

extern void rc_predict(int64_t T, const rc_fw_storage_t *X, rc_fw_storage_t *Y);

#ifndef T_TIME
#define T_TIME RC_FW_T
#endif

#define SYST_CSR (*(volatile uint32_t *)0xE000E010)
#define SYST_RVR (*(volatile uint32_t *)0xE000E014)
#define SYST_CVR (*(volatile uint32_t *)0xE000E018)
#define SYST_COUNTFLAG (1u << 16)

static rc_fw_storage_t Y[RC_FW_T * RC_FW_M];

static inline void sh_puts(const char *s) {
    register int r0 __asm__("r0") = 0x04;          /* SYS_WRITE0 */
    register const char *r1 __asm__("r1") = s;
    __asm__ volatile("bkpt #0xab" : "+r"(r0) : "r"(r1) : "memory");
}

__attribute__((noreturn)) static void sh_exit(int code) {
    register int r0 __asm__("r0") = 0x18;          /* SYS_EXIT */
    register int r1 __asm__("r1") = (code == 0) ? 0x20026 : code;
    __asm__ volatile("bkpt #0xab" : "+r"(r0) : "r"(r1) : "memory");
    while (1) { }
}

static void put_kv(const char *k, int32_t v) {
    char tmp[12], out[13];
    int n = 0, j = 0;
    sh_puts(k);
    if (v < 0) { sh_puts("-"); v = -v; }
    if (v == 0) tmp[n++] = '0';
    else while (v > 0) { tmp[n++] = (char)('0' + v % 10); v /= 10; }
    while (n > 0) out[j++] = tmp[--n];
    out[j] = 0;
    sh_puts(out);
    sh_puts("\n");
}

int main(void) {
    SYST_RVR = 0xFFFFFF;
    SYST_CVR = 0;
    SYST_CSR = 0x5;                 /* ENABLE | CLKSOURCE=processor clock */
    (void)SYST_CSR;                 /* clear any stale COUNTFLAG */
    uint32_t before = SYST_CVR;
    rc_predict((int64_t)T_TIME, g_x, Y);
    uint32_t after = SYST_CVR;
    uint32_t wrapped = SYST_CSR & SYST_COUNTFLAG;
    SYST_CSR = 0;

    int bad = 0;
    for (int i = 0; i < T_TIME * RC_FW_M; i++) {
        double d = (double)Y[i] - (double)g_y_ref[i];
        if (d < 0) d = -d;
        if (d > RC_FW_EPS) bad++;
    }
    sh_puts(bad == 0 ? "PARITY_OK\n" : "PARITY_FAIL\n");

    if (wrapped) {
        put_kv("ticks_per_step: ", -1);
    } else {
        uint32_t ticks = (before - after) & 0xFFFFFF;
        put_kv("ticks_per_step: ", (int32_t)(ticks / (uint32_t)T_TIME));
    }

    sh_puts("EMULATOR_EXIT\n");
    sh_exit(0);
    return 0;
}
