/* Affine (asymmetric per-tensor) quantized rc_predict demo.
 *
 * Pure integer kernel — storage_t is int8_t or int16_t. Both the input
 * samples and the reference outputs are embedded as quantized integers,
 * so the comparison on-device is also pure integer (no soft-FP, no libm).
 *
 * Template placeholders (filled in by CortexM0Target.compile_affine_quantized):
 *   @@T_LEN@@         — number of inference steps
 *   @@STORAGE_T@@     — int8_t / int16_t
 *   @@LUT_KIND@@      — "direct" / "linear_interp" / "polynomial"
 *   @@X_VALUES_Q@@    — comma-separated input samples (at input_scale)
 *   @@Y_VALUES_Q@@    — comma-separated reference outputs (at output_scale)
 */
#include <stdint.h>

#define T_LEN        @@T_LEN@@
typedef @@STORAGE_T@@ storage_t;

static const storage_t X_q[T_LEN]           = { @@X_VALUES_Q@@ };
static const storage_t Y_reference_q[T_LEN] = { @@Y_VALUES_Q@@ };

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
 * Integer-only decimal formatter (signed).                                *
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

/* ----------------------------------------------------------------------- *
 * main                                                                    *
 * ----------------------------------------------------------------------- */

int main(void)
{
    storage_t X[T_LEN];
    storage_t Y[T_LEN] = {0};
    char buf[40];

    for (int i = 0; i < T_LEN; i++) X[i] = X_q[i];

    sh_puts("==========================================\n");
    sh_puts("rc_predict (affine, storage=" "@@STORAGE_T@@"
            ", lut=" "@@LUT_KIND@@" ") on micro:bit\n");
    sh_puts("==========================================\n");

    rc_predict((int64_t)T_LEN, X, Y);

    int32_t max_abs_diff = 0;
    for (int t = 0; t < T_LEN; t++) {
        int32_t d = (int32_t)Y[t] - (int32_t)Y_reference_q[t];
        int32_t ad = (d < 0) ? -d : d;
        if (ad > max_abs_diff) max_abs_diff = ad;

        sh_puts("Step ");
        fmt_int(buf, t); sh_puts(buf);
        sh_puts(": X_q="); fmt_int(buf, (int32_t)X[t]); sh_puts(buf);
        sh_puts("  Y_ref="); fmt_int(buf, (int32_t)Y_reference_q[t]); sh_puts(buf);
        sh_puts("  Y="); fmt_int(buf, (int32_t)Y[t]); sh_puts(buf);
        sh_puts("\n");
    }

    sh_puts("------------------------------------------\n");
    sh_puts("Max |Y - Y_ref| (storage units): ");
    fmt_int(buf, max_abs_diff); sh_puts(buf);
    sh_puts("\n------------------------------------------\n");
    sh_puts("EMULATOR_EXIT\n");
    sh_exit(0);
    return 0;
}
