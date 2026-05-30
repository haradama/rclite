/* Naive hand-written *quantized* reservoir-computer kernel (mirage-style).
 *
 * Equivalent in spirit to what someone writing scratch i32 fixed-point C
 * would produce, mirroring mirage's `fixed_mul_explicit_shift` and LUT
 * tanh:
 *   - i32 storage everywhere; products promote to i64 then ashr+trunc
 *   - libm tanhf is replaced by a precomputed LUT with linear interpolation
 *   - 3-loop matmul; no SIMD intrinsics; no structural specialization
 *
 * Function signature matches rclite's quantized emit:
 *     void rc_predict(int64_t T, int32_t *X, int32_t *Y);
 * where X is preprocessed-and-quantized input at input_scale and Y is
 * output at state_scale.
 *
 * Template placeholders (filled by compare_host_full.py):
 *   @@N@@ @@K@@ @@M@@ @@F@@        dimensions
 *   @@STATE_FRAC@@ @@INPUT_FRAC@@ @@WEIGHT_FRAC@@  Q-format fractional bits
 *   @@LEAK_Q@@ @@ONE_MINUS_LEAK_Q@@ @@BIAS_Q@@     quantized step params
 *   @@INCLUDE_BIAS@@ @@INCLUDE_INPUT@@             readout config
 *   @@LUT_N@@ @@LUT_XMIN_Q@@ @@LUT_XMAX_Q@@        LUT geometry
 *   @@W_IN_VALUES_Q@@ @@W_RES_VALUES_Q@@ @@W_OUT_VALUES_Q@@   i32 weights
 *   @@LUT_TABLE_Q@@                                 i32 LUT entries
 */
#include <stdint.h>

#define N             @@N@@
#define K             @@K@@
#define M             @@M@@
#define F             @@F@@
#define STATE_FRAC    @@STATE_FRAC@@
#define INPUT_FRAC    @@INPUT_FRAC@@
#define WEIGHT_FRAC   @@WEIGHT_FRAC@@
#define LEAK_Q        @@LEAK_Q@@
#define ONE_MINUS_LEAK_Q @@ONE_MINUS_LEAK_Q@@
#define BIAS_Q        @@BIAS_Q@@
#define INCLUDE_BIAS  @@INCLUDE_BIAS@@
#define INCLUDE_INPUT @@INCLUDE_INPUT@@
#define LUT_N         @@LUT_N@@
#define LUT_XMIN_Q    @@LUT_XMIN_Q@@
#define LUT_XMAX_Q    @@LUT_XMAX_Q@@

#define SHIFT_IN   (WEIGHT_FRAC + INPUT_FRAC - STATE_FRAC)
#define SHIFT_RES  (WEIGHT_FRAC)

static const int32_t W_in_q[N * K]  = { @@W_IN_VALUES_Q@@ };
static const int32_t W_res_q[N * N] = { @@W_RES_VALUES_Q@@ };
static const int32_t W_out_q[M * F] = { @@W_OUT_VALUES_Q@@ };
static const int32_t lut_table_q[LUT_N] = { @@LUT_TABLE_Q@@ };

static inline int32_t fixed_mul_i32(int32_t a, int32_t b, int shift)
{
    int64_t prod = (int64_t)a * (int64_t)b;
    return (int32_t)(prod >> shift);
}

static int32_t tanh_lut_i32(int32_t x_q)
{
    if (x_q < LUT_XMIN_Q) x_q = LUT_XMIN_Q;
    if (x_q > LUT_XMAX_Q) x_q = LUT_XMAX_Q;
    int64_t num = (int64_t)(x_q - LUT_XMIN_Q);
    int32_t t_q = (int32_t)((num << STATE_FRAC) / (LUT_XMAX_Q - LUT_XMIN_Q));
    int32_t pos_q = t_q * (LUT_N - 1);
    int32_t i0 = pos_q >> STATE_FRAC;
    if (i0 < 0) i0 = 0;
    if (i0 > LUT_N - 2) i0 = LUT_N - 2;
    int32_t frac_q = pos_q - (i0 << STATE_FRAC);
    int32_t y0 = lut_table_q[i0];
    int32_t y1 = lut_table_q[i0 + 1];
    int32_t dy = y1 - y0;
    return y0 + fixed_mul_i32(dy, frac_q, STATE_FRAC);
}

void rc_predict(int64_t T, int32_t *X, int32_t *Y)
{
    int32_t h[N];
    int32_t pre[N];
    int32_t phi[F];

    for (int i = 0; i < N; i++) h[i] = 0;

    for (int64_t t = 0; t < T; t++) {
        /* pre[i] = bias + W_in @ X[t] + W_res @ h  (fixed_mul + i32 acc) */
        for (int i = 0; i < N; i++) {
            int32_t acc = BIAS_Q;
            for (int k = 0; k < K; k++) {
                acc += fixed_mul_i32(W_in_q[i*K + k], X[t*K + k], SHIFT_IN);
            }
            for (int j = 0; j < N; j++) {
                acc += fixed_mul_i32(W_res_q[i*N + j], h[j], SHIFT_RES);
            }
            pre[i] = acc;
        }

        /* h = (1-leak)*h + leak*tanh(pre)  via LUT */
        for (int i = 0; i < N; i++) {
            int32_t act = tanh_lut_i32(pre[i]);
            int32_t t1 = fixed_mul_i32(h[i], ONE_MINUS_LEAK_Q, STATE_FRAC);
            int32_t t2 = fixed_mul_i32(act, LEAK_Q, STATE_FRAC);
            h[i] = t1 + t2;
        }

        /* phi = [state_scale?] ++ [u?] ++ h  (mirage mixed-scale convention) */
        int off = 0;
#if INCLUDE_BIAS
        phi[off++] = (int32_t)1 << STATE_FRAC;
#endif
#if INCLUDE_INPUT
        for (int k = 0; k < K; k++) phi[off + k] = X[t*K + k];
        off += K;
#endif
        for (int i = 0; i < N; i++) phi[off + i] = h[i];

        /* readout: i64 acc, shift by state_frac, trunc to i32 */
        for (int m_i = 0; m_i < M; m_i++) {
            int64_t y_acc = 0;
            for (int f_i = 0; f_i < F; f_i++) {
                y_acc += (int64_t)W_out_q[m_i*F + f_i] * (int64_t)phi[f_i];
            }
            Y[t*M + m_i] = (int32_t)(y_acc >> STATE_FRAC);
        }
    }
}
