/* Game Boy Advance (ARM7TDMI / thumbv4t) cycle + parity harness for the
 * LLVM-cross-compiled rc_predict (float OR integer storage).
 *
 * The GBA has no semihosting; timing uses the hardware timers TM0+TM1
 * cascaded into a 32-bit cycle counter at the system clock (prescaler F/1),
 * and results go out over mGBA's debug log. Under mGBA the timer count is
 * DETERMINISTIC run-to-run (an emulated op-count proxy, not silicon cycles).
 * Code executes from cartridge ROM (with the GBA's ROM waitstates), as a real
 * cartridge would. The runner greps the log and stops the emulator on a
 * timeout (the GBA has no clean exit, so we spin afterwards).
 *
 * rc_data.h (benchmarks/_perf_kernels.py:emit_c_data_h) provides
 * rc_fw_storage_t, RC_FW_EPS, RC_FW_T/K/M, g_x[], g_y_ref[], and defines
 * RC_IS_FLOAT for the float build.
 */
#include <stdint.h>
#include "mgba_log.h"
#include "rc_data.h"

extern void rc_predict(int64_t T, const rc_fw_storage_t *X, rc_fw_storage_t *Y);

#ifndef T_TIME
#define T_TIME RC_FW_T
#endif

#define TM0CNT_L (*(volatile uint16_t *)0x4000100)
#define TM0CNT_H (*(volatile uint16_t *)0x4000102)
#define TM1CNT_L (*(volatile uint16_t *)0x4000104)
#define TM1CNT_H (*(volatile uint16_t *)0x4000106)

static rc_fw_storage_t Y[RC_FW_T * RC_FW_M];

/* Read the cascaded 32-bit counter, retrying across a low-half overflow. */
static uint32_t read_cycles(void) {
    uint16_t hi, lo, hi2;
    do { hi = TM1CNT_L; lo = TM0CNT_L; hi2 = TM1CNT_L; } while (hi != hi2);
    return ((uint32_t)hi << 16) | lo;
}

int main(void) {
    mgba_open();

    TM0CNT_H = 0; TM1CNT_H = 0;       /* disable */
    TM0CNT_L = 0; TM1CNT_L = 0;       /* reload 0 */
    TM1CNT_H = 0x84;                  /* enable | count-up (cascade on TM0) */
    TM0CNT_H = 0x80;                  /* enable | prescaler F/1 — starts both */
    uint32_t c0 = read_cycles();
    rc_predict((int64_t)T_TIME, g_x, Y);
    uint32_t c1 = read_cycles();
    TM0CNT_H = 0;                     /* stop */
    uint32_t ticks = c1 - c0;

    int bad = 0;
    for (int i = 0; i < T_TIME * RC_FW_M; i++) {
#ifdef RC_IS_FLOAT
        float d = (float)Y[i] - (float)g_y_ref[i];
        if (d < 0) d = -d;
        if (d > (float)RC_FW_EPS) bad++;
#else
        if ((int64_t)Y[i] != (int64_t)g_y_ref[i]) bad++;   /* exact */
#endif
    }

    ln_reset();
    ln_puts("ticks_per_step: ");
    ln_int((int32_t)(ticks / (uint32_t)T_TIME));
    ln_flush();
    mgba_log(bad == 0 ? "PARITY_OK" : "PARITY_FAIL");
    mgba_log(bad == 0 ? "TEST_PASS" : "TEST_FAIL");

    for (;;) { /* spin: the runner's timeout stops the emulator */ }
    return 0;
}
