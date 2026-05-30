/* QEMU micro:bit demo for the cross-compiled rc_predict.
 *
 * Compares the on-device (Cortex-M0, f32) prediction against an f64 host
 * reference embedded at build time. Output is written via ARM semihosting
 * (SYS_WRITE0, op 0x04). The program exits via SYS_EXIT (op 0x18) with
 * ADP_Stopped_ApplicationExit so qemu-system-arm terminates cleanly.
 *
 * Multi-input / multi-output aware: X is row-major (T_LEN x RC_INPUT_DIM),
 * Y row-major (T_LEN x RC_OUTPUT_DIM). The dims come from rc_predict.h.
 *
 * Template placeholders (filled in by CortexM0Target.compile):
 *   @@T_LEN@@      — number of inference steps T (not T*K / T*M)
 *   @@X_VALUES@@   — comma-separated input samples, row-major (T, K)
 *   @@Y_VALUES@@   — comma-separated host-reference predictions, (T, M)
 */
#include <stdint.h>
#include "rc_predict.h"

#define T_LEN @@T_LEN@@
#define X_LEN (T_LEN * RC_INPUT_DIM)
#define Y_LEN (T_LEN * RC_OUTPUT_DIM)

static const float X_in[X_LEN]        = { @@X_VALUES@@ };
static const float Y_reference[Y_LEN] = { @@Y_VALUES@@ };

/* ----------------------------------------------------------------------- *
 * Semihosting primitives (Cortex-M variant: BKPT #0xAB).                  *
 * ----------------------------------------------------------------------- */

/* The ARM semihosting BKPT may overwrite r0 with the syscall return value,
 * so we mark r0 as in-out ("+r"). Without this the compiler reuses the same
 * r0 value across consecutive sh_puts calls and the second BKPT lands with
 * garbage in r0. */
static inline void sh_puts(const char *s)
{
    register int r0 __asm__("r0") = 0x04;       /* SYS_WRITE0 */
    register const char *r1 __asm__("r1") = s;
    __asm__ volatile ("bkpt #0xab"
                      : "+r"(r0)
                      : "r"(r1)
                      : "memory");
}

__attribute__((noreturn))
static void sh_exit(int code)
{
    register int r0 __asm__("r0") = 0x18;       /* SYS_EXIT */
    register int r1 __asm__("r1") = (code == 0) ? 0x20026 : code;
    __asm__ volatile ("bkpt #0xab" : "+r"(r0) : "r"(r1) : "memory");
    while (1) { /* unreachable */ }
}

/* ----------------------------------------------------------------------- *
 * Tiny float / int formatters (positive bias OK; |v| < 10000).            *
 * ----------------------------------------------------------------------- */

static int fmt_int(char *buf, int v)
{
    char *p = buf;
    if (v < 0) { *p++ = '-'; v = -v; }
    char tmp[12];
    int n = 0;
    if (v == 0) {
        tmp[n++] = '0';
    } else {
        while (v > 0) { tmp[n++] = (char)('0' + v % 10); v /= 10; }
    }
    while (n > 0) *p++ = tmp[--n];
    *p = 0;
    return (int)(p - buf);
}

static int fmt_float(char *buf, float v, int decimals)
{
    char *p = buf;
    if (v < 0.0f) { *p++ = '-'; v = -v; }
    int whole = (int)v;
    float frac = v - (float)whole;

    char tmp[12];
    int n = 0;
    if (whole == 0) {
        tmp[n++] = '0';
    } else {
        while (whole > 0) { tmp[n++] = (char)('0' + whole % 10); whole /= 10; }
    }
    while (n > 0) *p++ = tmp[--n];

    if (decimals > 0) {
        *p++ = '.';
        for (int i = 0; i < decimals; i++) {
            frac *= 10.0f;
            int d = (int)frac;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            *p++ = (char)('0' + d);
            frac -= (float)d;
        }
    }
    *p = 0;
    return (int)(p - buf);
}

/* ----------------------------------------------------------------------- *
 * main                                                                    *
 * ----------------------------------------------------------------------- */

int main(void)
{
    float X[X_LEN];
    float Y[Y_LEN] = {0};
    char buf[32];

    sh_puts("=========================================\n");
    sh_puts("rc_predict on micro:bit (Cortex-M0, QEMU)\n");
    sh_puts("=========================================\n");

    for (int i = 0; i < X_LEN; i++) X[i] = X_in[i];

    rc_predict((int64_t)T_LEN, X, Y);

    /* Per-step display: print all RC_INPUT_DIM inputs and RC_OUTPUT_DIM
     * outputs (brackets collapse to a single value when the dim is 1). */
    for (int t = 0; t < T_LEN; t++) {
        sh_puts("Step ");
        fmt_int(buf, t);
        sh_puts(buf);
        sh_puts(": In=[");
        for (int k = 0; k < RC_INPUT_DIM; k++) {
            if (k) sh_puts(",");
            fmt_float(buf, X[t * RC_INPUT_DIM + k], 4);
            sh_puts(buf);
        }
        sh_puts("] Ref=[");
        for (int m = 0; m < RC_OUTPUT_DIM; m++) {
            if (m) sh_puts(",");
            fmt_float(buf, Y_reference[t * RC_OUTPUT_DIM + m], 4);
            sh_puts(buf);
        }
        sh_puts("] Pred=[");
        for (int m = 0; m < RC_OUTPUT_DIM; m++) {
            if (m) sh_puts(",");
            fmt_float(buf, Y[t * RC_OUTPUT_DIM + m], 4);
            sh_puts(buf);
        }
        sh_puts("]\n");
    }

    float sse = 0.0f;
    float max_abs_diff = 0.0f;
    for (int i = 0; i < Y_LEN; i++) {
        float d = Y[i] - Y_reference[i];
        sse += d * d;
        float ad = d < 0.0f ? -d : d;
        if (ad > max_abs_diff) max_abs_diff = ad;
    }
    float mse = sse / (float)Y_LEN;

    sh_puts("-----------------------------------------\n");
    sh_puts("MSE          : ");
    fmt_float(buf, mse, 6);
    sh_puts(buf);
    sh_puts("\nMax |diff|   : ");
    fmt_float(buf, max_abs_diff, 6);
    sh_puts(buf);
    sh_puts("\n-----------------------------------------\n");
    sh_puts("EMULATOR_EXIT\n");
    sh_exit(0);
    return 0;
}
