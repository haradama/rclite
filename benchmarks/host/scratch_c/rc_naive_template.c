/* Naive hand-written reservoir-computer kernel for benchmark comparison.
 *
 * "Naive" means: 3-loop dense matmul, libm tanh, no loop unrolling, no
 * structural specialization, no SIMD intrinsics. Compiled with
 * `gcc -O3 -lm`. This is what a first pass at writing a reservoir
 * computer in C looks like before any optimization work.
 *
 * The function signature matches what rclite emits:
 *     void rc_predict(int64_t T, double *X, double *Y);
 *
 * Template placeholders are filled by `benchmarks/compare_host_float.py`:
 *     @@N@@                 reservoir units
 *     @@K@@                 input dimension
 *     @@M@@                 output dimension
 *     @@F@@                 readout feature dimension (bias? + input? + state)
 *     @@LEAK@@              leak rate
 *     @@BIAS@@              reservoir bias scalar
 *     @@INPUT_OFFSET@@      input preprocess offset
 *     @@INPUT_SCALING@@     input preprocess scale
 *     @@INCLUDE_BIAS@@      0 or 1
 *     @@INCLUDE_INPUT@@     0 or 1
 *     @@W_IN_VALUES@@       comma-separated double constants, row-major (N*K)
 *     @@W_RES_VALUES@@      comma-separated, row-major (N*N) — sparse for SCR
 *     @@W_OUT_VALUES@@      comma-separated, row-major (M*F)
 */
#include <math.h>
#include <stdint.h>

#define N             @@N@@
#define K             @@K@@
#define M             @@M@@
#define F             @@F@@
#define LEAK          (@@LEAK@@)
#define ONE_MINUS_LEAK (1.0 - LEAK)
#define BIAS          (@@BIAS@@)
#define INPUT_OFFSET  (@@INPUT_OFFSET@@)
#define INPUT_SCALING (@@INPUT_SCALING@@)
#define INCLUDE_BIAS  @@INCLUDE_BIAS@@
#define INCLUDE_INPUT @@INCLUDE_INPUT@@

static const double W_in[N * K]  = { @@W_IN_VALUES@@ };
static const double W_res[N * N] = { @@W_RES_VALUES@@ };
static const double W_out[M * F] = { @@W_OUT_VALUES@@ };

void rc_predict(int64_t T, double *X, double *Y)
{
    double h[N];
    double pre[N];
    double phi[F];
    double u_pre[K];

    for (int i = 0; i < N; i++) h[i] = 0.0;

    for (int64_t t = 0; t < T; t++) {
        /* preprocess: u_pre = (X[t,:] - offset) * scale */
        for (int k = 0; k < K; k++) {
            u_pre[k] = (X[t * K + k] - INPUT_OFFSET) * INPUT_SCALING;
        }

        /* pre[i] = bias + W_in @ u_pre + W_res @ h  (naive 3-loop) */
        for (int i = 0; i < N; i++) {
            double acc = BIAS;
            for (int k = 0; k < K; k++) {
                acc += W_in[i * K + k] * u_pre[k];
            }
            for (int j = 0; j < N; j++) {
                acc += W_res[i * N + j] * h[j];
            }
            pre[i] = acc;
        }

        /* h = (1-leak)*h + leak*tanh(pre)  (libm tanh) */
        for (int i = 0; i < N; i++) {
            h[i] = ONE_MINUS_LEAK * h[i] + LEAK * tanh(pre[i]);
        }

        /* phi = [1?] ++ [u_raw?] ++ h */
        int off = 0;
#if INCLUDE_BIAS
        phi[off++] = 1.0;
#endif
#if INCLUDE_INPUT
        for (int k = 0; k < K; k++) phi[off + k] = X[t * K + k];
        off += K;
#endif
        for (int i = 0; i < N; i++) phi[off + i] = h[i];

        /* readout: Y[t, m] = sum_f W_out[m, f] * phi[f] */
        for (int m_i = 0; m_i < M; m_i++) {
            double y_acc = 0.0;
            for (int f_i = 0; f_i < F; f_i++) {
                y_acc += W_out[m_i * F + f_i] * phi[f_i];
            }
            Y[t * M + m_i] = y_acc;
        }
    }
}
