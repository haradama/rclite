/* rclite ESN firmware for the Arm Corstone-300 (Cortex-M55), CPU only.
 *
 * Same portable integer kernel as every other rclite target; here it runs on
 * the exact FVP subsystem ExecuTorch uses. Latency is measured with the M55
 * DWT cycle counter (CPU cycles), directly comparable to the ExecuTorch
 * runner's "Inference runtime: N CPU cycles". Output via semihosting.
 */
#include "syshelp.h"
#include "rc_data.h"   /* rc_fw_storage_t, RC_FW_T/K/M, g_x[], g_y_ref[] */

extern void rc_predict(int rc_T, const rc_fw_storage_t *X, rc_fw_storage_t *Y);

#ifndef NREP
#define NREP 50
#endif

/* Cortex-M55 DWT cycle counter (modeled by the FVP). */
#define DEMCR      (*(volatile unsigned *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile unsigned *)0xE0001000u)
#define DWT_CYCCNT (*(volatile unsigned *)0xE0001004u)

static rc_fw_storage_t Y[RC_FW_T * RC_FW_M];

int main(void) {
    sh_puts("==========================================\n");
    sh_puts("rclite reservoir (affine i8) on Corstone-300 (Cortex-M55, CPU)\n");
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

    /* latency in CPU cycles (DWT), NREP*T steps to amortize one-time costs */
    DEMCR |= (1u << 24);          /* TRCENA */
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1u;               /* CYCCNTENA */
    unsigned c0 = DWT_CYCCNT;
    for (int r = 0; r < NREP; r++) rc_predict(RC_FW_T, g_x, Y);
    unsigned c1 = DWT_CYCCNT;
    sh_put_kv("cpu_cycles_per_step: ",
              (int32_t)((c1 - c0) / ((unsigned)NREP * RC_FW_T)));

    sh_puts("EMULATOR_EXIT\n");
    sh_exit(0);
    return 0;
}
