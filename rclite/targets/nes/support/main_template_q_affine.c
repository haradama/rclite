/* Affine (asymmetric per-tensor) quantized rc_predict harness for the NES.
 *
 * Pure-integer kernel (rc_kernel.c, emitted by emit_affine_kernel_c) — the
 * 6502 has no hardware multiply and only ~2 KB of RAM, so the affine path
 * with a structured topology (SCR / DLR / DLRB) is the practical fit, exactly
 * as for the Arduino Uno. The weight / LUT tables are `const`, so llvm-mos
 * places them in PRG-ROM; only the small state buffers live in RAM.
 *
 * Reporting uses the de-facto NES test protocol (blargg): a result byte and a
 * NUL-terminated message string in PRG-RAM at $6000, validated by the
 * signature $DE $B0 $61 at $6001-$6003. Mesen's `--testrunner` watches $6000,
 * prints the $6004 message, and exits with the result byte (0 == pass). We
 * spin forever afterwards; the test runner stops the emulator.
 *
 * Template placeholders (filled in by NesTarget.compile_affine_quantized):
 *   @@T_STEPS@@       — number of inference steps (the `T` passed to rc_predict)
 *   @@X_LEN@@         — length of the embedded input array  (T * K)
 *   @@Y_LEN@@         — length of the embedded output array (T * M)
 *   @@STORAGE_T@@     — int8_t / int16_t
 *   @@PREDICT_T@@     — rc_predict T argument type (int32_t / int64_t)
 *   @@LUT_KIND@@      — "direct" / "linear_interp" / "polynomial"
 *   @@TOL@@           — max allowed |Y - Y_ref| (storage units) for TEST_PASS
 *   @@X_VALUES_Q@@    — comma-separated input samples  (at input_scale)
 *   @@Y_VALUES_Q@@    — comma-separated reference outputs (at output_scale)
 */
#include <stdint.h>
#include <ines.h>

/* Map 8 KB of PRG-RAM at $6000 (NROM "family" / iNES byte 10) so the blargg
 * test region below is backed by real RAM both in the linker layout and in the
 * emulator. No C variable is placed there (the kernel's state lives in the
 * 2 KB internal RAM), so $6000.. is free for the raw protocol writes. */
MAPPER_PRG_RAM_KB(8);

#define T_STEPS @@T_STEPS@@
#define X_LEN   @@X_LEN@@
#define Y_LEN   @@Y_LEN@@
#define TOL     @@TOL@@
typedef @@STORAGE_T@@ storage_t;
typedef @@PREDICT_T@@ predict_t;

extern void rc_predict(predict_t T, const storage_t *X, storage_t *Y);

static const storage_t X_q[X_LEN]           = { @@X_VALUES_Q@@ };
static const storage_t Y_reference_q[Y_LEN] = { @@Y_VALUES_Q@@ };
static storage_t Y_out[Y_LEN];

/* blargg test protocol region in PRG-RAM ($6000-) */
#define NES_TEST_STATUS (*(volatile uint8_t *)0x6000)
static volatile uint8_t *const NES_TEST_SIG = (volatile uint8_t *)0x6001;
static volatile char    *const NES_TEST_MSG = (volatile char *)0x6004;

static void test_msg(const char *s)
{
    volatile char *d = NES_TEST_MSG;
    while (*s) *d++ = *s++;
    *d = 0;
}

static void test_result(uint8_t code, const char *s)
{
    test_msg(s);
    NES_TEST_STATUS = code;   /* 0x00..0x7F: done, code is the exit status */
}

int main(void)
{
    int t;
    int32_t max_abs_diff = 0;

    /* Announce the blargg protocol so the test runner starts watching. */
    NES_TEST_SIG[0] = 0xDE;
    NES_TEST_SIG[1] = 0xB0;
    NES_TEST_SIG[2] = 0x61;
    NES_TEST_STATUS = 0x80;   /* 0x80: test running */

    rc_predict((predict_t)T_STEPS, X_q, Y_out);

    for (t = 0; t < Y_LEN; t++) {
        int32_t d  = (int32_t)Y_out[t] - (int32_t)Y_reference_q[t];
        int32_t ad = (d < 0) ? -d : d;
        if (ad > max_abs_diff) max_abs_diff = ad;
    }

    if (max_abs_diff <= TOL)
        test_result(0x00, "TEST_PASS\n");
    else
        test_result(0x01, "TEST_FAIL\n");

    for (;;) { /* spin: the test runner stops the emulator */ }
    return 0;
}
