/* rclite reservoir firmware for the micro:bit (Cortex-M0).
 *
 * Runs the portable integer kernel over an embedded Mackey-Glass test
 * sequence, checks bit-exactness against the host AffineQuantizedExecutor,
 * and times the kernel. Same startup.c / nrf51.ld / semihosting as the TFLM
 * firmware so Flash (text+data) and static RAM (data+bss) compare directly.
 */
#include "syshelp.h"
#include "rc_data.h"   /* rc_fw_storage_t, RC_FW_T/K/M, g_x[], g_y_ref[] */

extern void rc_predict(int rc_T, const rc_fw_storage_t *X, rc_fw_storage_t *Y);

#ifndef NREP
#define NREP 50        /* timing repetitions to amortize semihosting jitter */
#endif

static rc_fw_storage_t Y[RC_FW_T * RC_FW_M];

int main(void) {
    sh_puts("==========================================\n");
    sh_puts("rclite reservoir (affine int) on micro:bit (Cortex-M0)\n");
    sh_puts("==========================================\n");

    rc_predict(RC_FW_T, g_x, Y);

    int32_t max_abs_diff = 0;
    for (int i = 0; i < RC_FW_T * RC_FW_M; i++) {
        int32_t d = (int32_t)Y[i] - (int32_t)g_y_ref[i];
        if (d < 0) d = -d;
        if (d > max_abs_diff) max_abs_diff = d;
    }
    sh_put_kv("max_abs_diff_vs_host: ", max_abs_diff);
    sh_puts(max_abs_diff == 0 ? "PARITY_OK\n" : "PARITY_FAIL\n");

    /* latency: NREP runs of T steps; report instructions per single step. */
    uint64_t e0 = sh_elapsed();
    for (int r = 0; r < NREP; r++) rc_predict(RC_FW_T, g_x, Y);
    uint64_t e1 = sh_elapsed();
    sh_put_kv("instr_per_step: ", (int32_t)((e1 - e0) / ((uint64_t)NREP * RC_FW_T)));

    sh_puts("EMULATOR_EXIT\n");
    sh_exit(0);
    return 0;
}
