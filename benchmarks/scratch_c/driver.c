/* Standalone driver for callgrind / perf instruction counting.
 *
 * Usage:
 *     ./driver <path/to/librc.so> <T> <n_calls>
 *
 * dlopen()s the given shared library, looks up `rc_predict`, and calls
 * it `n_calls` times on a deterministic synthetic input. The first call
 * acts as a warmup (it dominates the dynamic loader's first-touch cost);
 * callgrind divides the remaining calls to give per-inference instruction
 * counts.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>

typedef void (*rc_predict_fn)(int64_t, double *, double *);

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
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 2;
    }
    rc_predict_fn fn = (rc_predict_fn)dlsym(handle, "rc_predict");
    if (!fn) {
        fprintf(stderr, "dlsym(rc_predict): %s\n", dlerror());
        dlclose(handle);
        return 3;
    }

    double *X = (double *)malloc((size_t)T * sizeof(double));
    double *Y = (double *)malloc((size_t)T * sizeof(double));
    if (!X || !Y) return 4;

    /* Deterministic synthetic input in the Mackey-Glass operating range. */
    for (int64_t i = 0; i < T; i++) {
        X[i] = 0.95 + 0.15 * sin((double)i * 0.05) + 0.05 * sin((double)i * 0.37);
    }

    /* Warmup — page-in code/data, populate caches */
    fn(T, X, Y);

    /* Measured region — callgrind sums instructions across these calls */
    for (int i = 0; i < n_calls; i++) {
        fn(T, X, Y);
    }

    /* Touch Y to prevent the compiler from optimizing away the work */
    double sink = 0.0;
    for (int64_t i = 0; i < T; i++) sink += Y[i];
    fprintf(stderr, "sink=%.5g\n", sink);

    free(X);
    free(Y);
    dlclose(handle);
    return 0;
}
