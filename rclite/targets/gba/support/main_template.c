/* Float (f32, soft-float) rc_predict demo for the GBA.
 *
 * ARMv4T has no FPU, so this path runs through libgcc soft-float and is slow;
 * the quantized/affine paths are preferred for the GBA. Compares the on-device
 * prediction against an embedded host reference. Output goes through mGBA's
 * debug log and the driver prints TEST_PASS / TEST_FAIL for the runner.
 *
 * Multi-input / multi-output aware: X is row-major (T_LEN x RC_INPUT_DIM),
 * Y row-major (T_LEN x RC_OUTPUT_DIM). The dims come from rc_predict.h.
 *
 * Template placeholders (filled in by GbaTarget.compile):
 *   @@T_LEN@@      — number of inference steps T (not T*K / T*M)
 *   @@TOLF@@       — max allowed |Y - Y_ref| for TEST_PASS
 *   @@X_VALUES@@   — comma-separated input samples, row-major (T, K)
 *   @@Y_VALUES@@   — comma-separated host-reference predictions, (T, M)
 */
#include <stdint.h>
#include "rc_predict.h"
#include "mgba_log.h"

#define T_LEN  @@T_LEN@@
#define TOLF   @@TOLF@@f
#define X_LEN  (T_LEN * RC_INPUT_DIM)
#define Y_LEN  (T_LEN * RC_OUTPUT_DIM)

static const float X_in[X_LEN]        = { @@X_VALUES@@ };
static const float Y_reference[Y_LEN] = { @@Y_VALUES@@ };

/* Append a float with `decimals` fractional digits to the mGBA line buffer. */
static void ln_float(float v, int decimals)
{
    char b[24]; int k = 0;
    if (v < 0.0f) { b[k++] = '-'; v = -v; }
    int whole = (int)v;
    float frac = v - (float)whole;
    char tmp[12]; int n = 0;
    if (whole == 0) tmp[n++] = '0';
    else while (whole > 0) { tmp[n++] = (char)('0' + whole % 10); whole /= 10; }
    while (n > 0) b[k++] = tmp[--n];
    if (decimals > 0) {
        b[k++] = '.';
        for (int i = 0; i < decimals; i++) {
            frac *= 10.0f;
            int d = (int)frac;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            b[k++] = (char)('0' + d);
            frac -= (float)d;
        }
    }
    b[k] = 0;
    ln_puts(b);
}

int main(void)
{
    float X[X_LEN];
    float Y[Y_LEN] = {0};

    for (int i = 0; i < X_LEN; i++) X[i] = X_in[i];

    mgba_open();
    mgba_log("rc_predict (f32, soft-float) on GBA");

    rc_predict((int64_t)T_LEN, X, Y);

    for (int t = 0; t < T_LEN; t++) {
        ln_reset();
        ln_puts("Step "); ln_int(t);
        ln_puts(": Ref=[");
        for (int m = 0; m < RC_OUTPUT_DIM; m++) {
            if (m) ln_puts(",");
            ln_float(Y_reference[t * RC_OUTPUT_DIM + m], 4);
        }
        ln_puts("] Pred=[");
        for (int m = 0; m < RC_OUTPUT_DIM; m++) {
            if (m) ln_puts(",");
            ln_float(Y[t * RC_OUTPUT_DIM + m], 4);
        }
        ln_puts("]");
        ln_flush();
    }

    float max_abs_diff = 0.0f;
    for (int i = 0; i < Y_LEN; i++) {
        float d  = Y[i] - Y_reference[i];
        float ad = (d < 0.0f) ? -d : d;
        if (ad > max_abs_diff) max_abs_diff = ad;
    }

    ln_reset(); ln_puts("max_abs_diff="); ln_float(max_abs_diff, 6); ln_flush();
    mgba_log((max_abs_diff <= TOLF) ? "TEST_PASS" : "TEST_FAIL");

    for (;;) { /* spin: the runner's timeout stops the emulator */ }
    return 0;
}
