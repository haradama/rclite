/* Quantized fixed-point demo for the cross-compiled rc_predict.
 *
 * The kernel takes storage_t inputs at input_scale and returns storage_t
 * outputs at state_scale. Float values are never touched on-device —
 * formatting is pure integer arithmetic. Compared against the host
 * reference embedded as state-scaled storage_t constants.
 *
 * Multi-input / multi-output aware: X_q is row-major (T_LEN x RC_K), the
 * reference Y row-major (T_LEN x RC_M).
 *
 * Template placeholders (filled in by CortexM0Target.compile_quantized):
 *   @@T_LEN@@        — number of inference steps T (not T*K / T*M)
 *   @@RC_K@@         — input dimension K
 *   @@RC_M@@         — output dimension M
 *   @@STATE_FRAC@@   — state Q-format fractional bits
 *   @@STORAGE_T@@    — kernel storage type (int8_t / int16_t / int32_t)
 *   @@X_VALUES_Q@@   — comma-separated input samples (at input_scale), (T, K)
 *   @@Y_VALUES_Q@@   — comma-separated reference outputs (at state_scale), (T, M)
 */
#include <stdint.h>

#define T_LEN          @@T_LEN@@
#define RC_K           @@RC_K@@
#define RC_M           @@RC_M@@
#define X_LEN          (T_LEN * RC_K)
#define Y_LEN          (T_LEN * RC_M)
#define STATE_FRAC     @@STATE_FRAC@@
#define STATE_SCALE    (1 << STATE_FRAC)

typedef @@STORAGE_T@@ storage_t;

static const storage_t X_q[X_LEN]           = { @@X_VALUES_Q@@ };
static const storage_t Y_reference_q[Y_LEN] = { @@Y_VALUES_Q@@ };

extern void rc_predict(int64_t T, storage_t *X, storage_t *Y);

/* ----------------------------------------------------------------------- *
 * Semihosting (Cortex-M variant: BKPT #0xAB).                             *
 * ----------------------------------------------------------------------- */

static inline void sh_puts(const char *s)
{
    register int r0 __asm__("r0") = 0x04;       /* SYS_WRITE0 */
    register const char *r1 __asm__("r1") = s;
    __asm__ volatile ("bkpt #0xab"
                      : "+r"(r0) : "r"(r1) : "memory");
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
 * Integer-only formatters (no libm / soft-float).                         *
 * ----------------------------------------------------------------------- */

static int fmt_int(char *buf, int32_t v)
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

/* Format a Q-format fixed-point value with `decimals` digits past the point. */
static int fmt_fixed(char *buf, int32_t v, int frac_bits, int decimals)
{
    char *p = buf;
    if (v < 0) { *p++ = '-'; v = -v; }
    int32_t scale = (int32_t)1 << frac_bits;
    int32_t whole = v >> frac_bits;
    int32_t frac = v & (scale - 1);

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
            frac *= 10;
            int d = frac >> frac_bits;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            *p++ = (char)('0' + d);
            frac -= ((int32_t)d) << frac_bits;
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
    storage_t X[X_LEN];
    storage_t Y[Y_LEN] = {0};
    char buf[40];

    for (int i = 0; i < X_LEN; i++) X[i] = X_q[i];

    sh_puts("==========================================\n");
    sh_puts("rc_predict (Q-format, storage=" "@@STORAGE_T@@" ") on micro:bit\n");
    sh_puts("STATE_FRAC = ");
    fmt_int(buf, STATE_FRAC);
    sh_puts(buf);
    sh_puts("\n==========================================\n");

    rc_predict((int64_t)T_LEN, X, Y);

    for (int t = 0; t < T_LEN; t++) {
        sh_puts("Step ");
        fmt_int(buf, t);
        sh_puts(buf);
        sh_puts(": X_q=[");
        for (int k = 0; k < RC_K; k++) {
            if (k) sh_puts(",");
            fmt_int(buf, (int32_t)X[t * RC_K + k]);
            sh_puts(buf);
        }
        sh_puts("] Y_ref=[");
        for (int m = 0; m < RC_M; m++) {
            if (m) sh_puts(",");
            fmt_fixed(buf, (int32_t)Y_reference_q[t * RC_M + m], STATE_FRAC, 4);
            sh_puts(buf);
        }
        sh_puts("] Y=[");
        for (int m = 0; m < RC_M; m++) {
            if (m) sh_puts(",");
            fmt_fixed(buf, (int32_t)Y[t * RC_M + m], STATE_FRAC, 4);
            sh_puts(buf);
        }
        sh_puts("]\n");
    }

    int32_t max_abs_diff = 0;
    for (int i = 0; i < Y_LEN; i++) {
        int32_t d = (int32_t)Y[i] - (int32_t)Y_reference_q[i];
        int32_t ad = (d < 0) ? -d : d;
        if (ad > max_abs_diff) max_abs_diff = ad;
    }

    sh_puts("------------------------------------------\n");
    sh_puts("Max |Y - Y_ref| (state-scale units): ");
    fmt_int(buf, max_abs_diff);
    sh_puts(buf);
    sh_puts("\n                          (decoded): ");
    fmt_fixed(buf, max_abs_diff, STATE_FRAC, 6);
    sh_puts(buf);
    sh_puts("\n------------------------------------------\n");
    sh_puts("EMULATOR_EXIT\n");
    sh_exit(0);
    return 0;
}
