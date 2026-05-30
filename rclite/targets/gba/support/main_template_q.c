/* Symmetric (Q-format) quantized rc_predict demo for the GBA.
 *
 * The kernel takes storage_t inputs at input_scale and returns storage_t
 * outputs at state_scale. Pure integer arithmetic on-device; output goes
 * through mGBA's debug log and the driver prints TEST_PASS / TEST_FAIL.
 *
 * Multi-input / multi-output aware: X_q is row-major (T_LEN x RC_K), the
 * reference Y row-major (T_LEN x RC_M).
 *
 * Template placeholders (filled in by GbaTarget.compile_quantized):
 *   @@T_LEN@@        — number of inference steps T (not T*K / T*M)
 *   @@RC_K@@         — input dimension K
 *   @@RC_M@@         — output dimension M
 *   @@STATE_FRAC@@   — state Q-format fractional bits
 *   @@STORAGE_T@@    — kernel storage type (int8_t / int16_t / int32_t)
 *   @@TOL@@          — max allowed |Y - Y_ref| (state-scale units) for TEST_PASS
 *   @@X_VALUES_Q@@   — comma-separated input samples (at input_scale), (T, K)
 *   @@Y_VALUES_Q@@   — comma-separated reference outputs (at state_scale), (T, M)
 */
#include <stdint.h>
#include "mgba_log.h"

#define T_LEN       @@T_LEN@@
#define RC_K        @@RC_K@@
#define RC_M        @@RC_M@@
#define X_LEN       (T_LEN * RC_K)
#define Y_LEN       (T_LEN * RC_M)
#define STATE_FRAC  @@STATE_FRAC@@
#define TOL         @@TOL@@
typedef @@STORAGE_T@@ storage_t;

static const storage_t X_q[X_LEN]           = { @@X_VALUES_Q@@ };
static const storage_t Y_reference_q[Y_LEN] = { @@Y_VALUES_Q@@ };

extern void rc_predict(int64_t T, storage_t *X, storage_t *Y);

int main(void)
{
    storage_t X[X_LEN];
    storage_t Y[Y_LEN] = {0};

    for (int i = 0; i < X_LEN; i++) X[i] = X_q[i];

    mgba_open();
    mgba_log("rc_predict (Q-format, storage=" "@@STORAGE_T@@" ") on GBA");

    rc_predict((int64_t)T_LEN, X, Y);

    for (int t = 0; t < T_LEN; t++) {
        ln_reset();
        ln_puts("Step "); ln_int(t);
        ln_puts(": X_q=[");
        for (int k = 0; k < RC_K; k++) {
            if (k) ln_puts(",");
            ln_int((int32_t)X[t * RC_K + k]);
        }
        ln_puts("] Y_ref=[");
        for (int m = 0; m < RC_M; m++) {
            if (m) ln_puts(",");
            ln_fixed((int32_t)Y_reference_q[t * RC_M + m], STATE_FRAC, 4);
        }
        ln_puts("] Y=[");
        for (int m = 0; m < RC_M; m++) {
            if (m) ln_puts(",");
            ln_fixed((int32_t)Y[t * RC_M + m], STATE_FRAC, 4);
        }
        ln_puts("]");
        ln_flush();
    }

    int32_t max_abs_diff = 0;
    for (int i = 0; i < Y_LEN; i++) {
        int32_t d  = (int32_t)Y[i] - (int32_t)Y_reference_q[i];
        int32_t ad = (d < 0) ? -d : d;
        if (ad > max_abs_diff) max_abs_diff = ad;
    }

    ln_reset(); ln_puts("max_abs_diff="); ln_int(max_abs_diff); ln_flush();
    mgba_log((max_abs_diff <= TOL) ? "TEST_PASS" : "TEST_FAIL");

    for (;;) { /* spin: the runner's timeout stops the emulator */ }
    return 0;
}
