/* Affine (asymmetric per-tensor) quantized rc_predict demo for the GBA.
 *
 * Pure integer kernel — storage_t is int8_t or int16_t. Both the input
 * samples and the reference outputs are embedded as quantized integers, so
 * the comparison on-device is also pure integer. Output goes through mGBA's
 * debug log; the driver prints TEST_PASS / TEST_FAIL for the runner to grep.
 *
 * Multi-input / multi-output aware: X_q is row-major (T_LEN x RC_K), the
 * reference Y row-major (T_LEN x RC_M).
 *
 * Template placeholders (filled in by GbaTarget.compile_affine_quantized):
 *   @@T_LEN@@         — number of inference steps T (not T*K / T*M)
 *   @@RC_K@@          — input dimension K
 *   @@RC_M@@          — output dimension M
 *   @@STORAGE_T@@     — int8_t / int16_t
 *   @@LUT_KIND@@      — "direct" / "linear_interp" / "polynomial"
 *   @@TOL@@           — max allowed |Y - Y_ref| (storage units) for TEST_PASS
 *   @@X_VALUES_Q@@    — comma-separated input samples (at input_scale), (T, K)
 *   @@Y_VALUES_Q@@    — comma-separated reference outputs (at output_scale), (T, M)
 */
#include <stdint.h>
#include "mgba_log.h"

#define T_LEN  @@T_LEN@@
#define RC_K   @@RC_K@@
#define RC_M   @@RC_M@@
#define X_LEN  (T_LEN * RC_K)
#define Y_LEN  (T_LEN * RC_M)
#define TOL    @@TOL@@
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
    mgba_log("rc_predict (affine, storage=" "@@STORAGE_T@@"
             ", lut=" "@@LUT_KIND@@" ") on GBA");

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
            ln_int((int32_t)Y_reference_q[t * RC_M + m]);
        }
        ln_puts("] Y=[");
        for (int m = 0; m < RC_M; m++) {
            if (m) ln_puts(",");
            ln_int((int32_t)Y[t * RC_M + m]);
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
