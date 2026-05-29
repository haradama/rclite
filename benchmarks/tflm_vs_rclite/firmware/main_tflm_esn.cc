/* TFLM firmware running the SAME ESN as rclite, as a single-step cell with
 * external float state feedback (micro:bit / Cortex-M0).
 *
 * Identical reservoir to the rclite firmware (same float weights) — this
 * isolates the deployment stack: TFLM interpreter vs rclite codegen. TFLM has
 * no reservoir op, so the recurrence is a per-step Invoke and W_res is a dense
 * 80x80. Same startup/linker/semihosting as the other firmwares.
 */
#include "syshelp.h"
#include "model_esn_data.h"     /* g_esn_model[], g_esn_model_len            */
#include "esn_test_data.h"      /* ESN_T, ESN_N, ESN_YSCALE, g_esn_x[], ref  */

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef ARENA_SIZE
#define ARENA_SIZE 8192
#endif
alignas(16) static uint8_t g_arena[ARENA_SIZE];

#ifndef NREP
#define NREP 100
#endif

static TfLiteTensor *by_len(tflite::MicroInterpreter &I, bool input, int len) {
    int n = input ? 1 /*we know 2*/ : 1;
    (void)n;
    int cnt = input ? 2 : 2;
    for (int i = 0; i < cnt; i++) {
        TfLiteTensor *t = input ? I.input(i) : I.output(i);
        if (t->dims->data[t->dims->size - 1] == len) return t;
    }
    return nullptr;
}

int main(void) {
    sh_puts("==========================================\n");
    sh_puts("TFLM ESN cell (same reservoir as rclite) on micro:bit (Cortex-M0)\n");
    sh_puts("==========================================\n");

    const tflite::Model *model = tflite::GetModel(g_esn_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        sh_puts("ERR: schema mismatch\n"); sh_exit(1);
    }
    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddFullyConnected();
    resolver.AddTanh();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddConcatenation();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter interp(model, resolver, g_arena, ARENA_SIZE);
    if (interp.AllocateTensors() != kTfLiteOk) {
        sh_puts("ERR: AllocateTensors failed (arena too small?)\n"); sh_exit(1);
    }
    sh_put_kv("arena_used_bytes: ", (int32_t)interp.arena_used_bytes());

    TfLiteTensor *xin = by_len(interp, true, 1);
    TfLiteTensor *hin = by_len(interp, true, ESN_N);
    TfLiteTensor *yout = by_len(interp, false, 1);
    TfLiteTensor *hout = by_len(interp, false, ESN_N);
    if (!xin || !hin || !yout || !hout) { sh_puts("ERR: tensor map\n"); sh_exit(1); }

    /* ---- correctness: loop with state feedback, compare to host ----
     * start from the host's warmed-up reservoir state so both replay the
     * exact same trajectory (the readout is chaotic-sensitive to h). */
    static float h[ESN_N];
    for (int i = 0; i < ESN_N; i++) h[i] = g_esn_h0[i];
    int32_t max_abs_diff = 0;
    for (int t = 0; t < ESN_T; t++) {
        xin->data.f[0] = g_esn_x[t];
        for (int i = 0; i < ESN_N; i++) hin->data.f[i] = h[i];
        if (interp.Invoke() != kTfLiteOk) { sh_puts("ERR: Invoke\n"); sh_exit(1); }
        for (int i = 0; i < ESN_N; i++) h[i] = hout->data.f[i];
        int32_t yq = (int32_t)(yout->data.f[0] * (float)ESN_YSCALE
                               + (yout->data.f[0] >= 0 ? 0.5f : -0.5f));
        int32_t d = yq - g_esn_yref_scaled[t];
        if (d < 0) d = -d;
        if (d > max_abs_diff) max_abs_diff = d;
    }
    /* The cell has a float I/O boundary; over a chaotic recurrent loop, x86
     * vs ARM float32 ULP differences drift apart (rclite's pure-integer kernel
     * is bit-exact by contrast). Accept a functional match well inside the
     * int8 quantization error (~0.35*signal); report the actual drift. */
    sh_put_kv("max_abs_diff_scaled: ", max_abs_diff);   /* units of 1/ESN_YSCALE */
    sh_puts(max_abs_diff <= 2000 ? "FUNCTIONAL_MATCH\n" : "DIVERGED\n");

    /* ---- latency: NREP single-step invokes ---- */
    for (int i = 0; i < ESN_N; i++) hin->data.f[i] = 0.0f;
    xin->data.f[0] = g_esn_x[0];
    uint64_t e0 = sh_elapsed();
    for (int r = 0; r < NREP; r++) interp.Invoke();
    uint64_t e1 = sh_elapsed();
    sh_put_kv("instr_per_step: ", (int32_t)((e1 - e0) / (uint64_t)NREP));

    sh_puts("EMULATOR_EXIT\n");
    sh_exit(0);
    return 0;
}

extern "C" void __cxa_pure_virtual(void) { while (1) {} }
extern "C" int __cxa_atexit(void (*)(void *), void *, void *) { return 0; }
void *__dso_handle = 0;
