/* Standalone driver for callgrind instruction counting of the quantized
 * kernel.
 *
 * Usage:
 *     ./driver_q <path/to/lib.so> <T> <n_calls>
 *
 * The quantized rc_predict's signature is
 *     void rc_predict(int64_t T, int32_t *X, int32_t *Y);
 * so this driver fills X with a deterministic Q.16-ish pattern (the kernel
 * does the same arithmetic regardless of input *values*, only branch in
 * the LUT clamp differs; both bounds are exercised by the sin pattern).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>

typedef void (*rc_predict_q_fn)(int64_t, int32_t *, int32_t *);

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <lib.so> <T> <n_calls>\n", argv[0]);
        return 1;
    }
    const char *so_path = argv[1];
    int64_t T = (int64_t)atoll(argv[2]);
    int n_calls = atoi(argv[3]);

    void *handle = dlopen(so_path, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    rc_predict_q_fn fn = (rc_predict_q_fn)dlsym(handle, "rc_predict");
    if (!fn) {
        fprintf(stderr, "dlsym: %s\n", dlerror());
        dlclose(handle);
        return 3;
    }

    int32_t *X = (int32_t *)malloc((size_t)T * sizeof(int32_t));
    int32_t *Y = (int32_t *)malloc((size_t)T * sizeof(int32_t));
    if (!X || !Y) return 4;

    /* Q.16-ish deterministic input — values around 1.0 with oscillation. */
    for (int64_t i = 0; i < T; i++) {
        double v = 0.95 + 0.15 * sin((double)i * 0.05);
        X[i] = (int32_t)(v * 65536.0);
    }

    fn(T, X, Y);  /* warmup */
    for (int i = 0; i < n_calls; i++) fn(T, X, Y);

    int64_t sink = 0;
    for (int64_t i = 0; i < T; i++) sink += Y[i];
    fprintf(stderr, "sink=%lld\n", (long long)sink);

    free(X);
    free(Y);
    dlclose(handle);
    return 0;
}
