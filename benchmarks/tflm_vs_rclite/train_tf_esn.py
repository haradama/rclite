"""Rebuild the *same* float ESN (from out/esn_params.npz) as a TFLite
single-step cell, int8-quantize it, and run it through the TFLite interpreter
in a loop with state feedback on the same Mackey-Glass targets.

This is the apples-to-apples model for the TFLM-vs-rclite comparison: an
identical reservoir, deployed two ways (TFLM interpreter vs rclite codegen).
TFLM has no reservoir/RNN-cell op, so the recurrence is expressed as a cell
(FullyConnected x3 + Tanh + Mul/Add) invoked per step with external state;
W_res is necessarily a dense 80x80 (SCR structure can't be expressed).

Run with the TF venv (after export_esn_params.py):
    /tmp/tfenv/bin/python benchmarks/tflm_vs_rclite/train_tf_esn.py
"""
from __future__ import annotations
import json
import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
import tensorflow as tf  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
FW = HERE / "firmware"
T_FW = 200          # firmware test-sequence length


def build_cell(p):
    """Keras single-step ESN cell: (x_raw, h_prev) -> (h_t, y).

    Matches rclite exactly: the reservoir sees the *preprocessed* input
    u = (x - offset) * scaling, but the readout pass-through uses the *raw*
    input x (see RCExecutor._augment). Centering is done inside the cell so
    the loop feeds raw samples.
    """
    N, K, M = int(p["N"]), int(p["K"]), int(p["M"])
    leak = float(p["leak"])
    W_in, W_res = p["W_in"], p["W_res"]                  # (N,K), (N,N)
    res_bias = float(p["res_bias"])
    W_out_input, W_out_state = p["W_out_input"], p["W_out_state"]
    W_out_bias = p["W_out_bias"]
    off, scaling = float(p["input_offset"]), float(p["input_scaling"])

    x = tf.keras.Input(shape=(K,), name="x")            # raw input
    h_prev = tf.keras.Input(shape=(N,), name="h_prev")
    # u = (x - off) * scaling  == x*scaling + (-off*scaling)
    u = tf.keras.layers.Rescaling(scaling, offset=-off * scaling)(x)
    z = tf.keras.layers.Concatenate()([u, h_prev])      # (K+N,)
    pre = tf.keras.layers.Dense(N, name="pre")(z)
    act = tf.keras.layers.Activation("tanh")(pre)
    h_t = tf.keras.layers.Add(name="h")([
        tf.keras.layers.Rescaling(1.0 - leak)(h_prev),
        tf.keras.layers.Rescaling(leak)(act),
    ])
    z2 = tf.keras.layers.Concatenate()([x, h_t])        # RAW x for readout
    y = tf.keras.layers.Dense(M, name="yout")(z2)
    model = tf.keras.Model([x, h_prev], [h_t, y])

    # pre kernel (K+N, N): rows 0..K-1 = W_in, rows K.. = W_res^T
    pre_kernel = np.zeros((K + N, N), np.float32)
    pre_kernel[:K, :] = W_in.T
    pre_kernel[K:, :] = W_res.T
    model.get_layer("pre").set_weights([pre_kernel,
                                        np.full(N, res_bias, np.float32)])
    # yout kernel (K+N, M): rows 0..K-1 = W_out_input^T, rows K.. = W_out_state^T
    yk = np.zeros((K + N, M), np.float32)
    if W_out_input.shape[1] == K:
        yk[:K, :] = W_out_input.T
    yk[K:, :] = W_out_state.T
    ybias = (W_out_bias.reshape(-1).astype(np.float32)
             if W_out_bias.size else np.zeros(M, np.float32))
    model.get_layer("yout").set_weights([yk, ybias])
    return model, N, K, M


def run_float(model, X, N):
    """Loop the float cell over inputs X (T,K); return predictions (T,M)."""
    T = X.shape[0]
    h = np.zeros((1, N), np.float32)
    preds = []
    for t in range(T):
        h, y = model([X[t:t + 1].astype(np.float32), h], training=False)
        h = h.numpy()
        preds.append(float(np.asarray(y).ravel()[0]))
    return np.array(preds)


def main() -> int:
    p = np.load(OUT / "esn_params.npz")
    model, N, K, M = build_cell(p)

    s = common.series().astype(np.float32)
    Xseq = s[:-1, None]                   # RAW input; the cell centers internally
    train_t, test_t = common.target_indices()
    n_fit = common.TRAIN_END

    # representative (x, h_prev) pairs from a float run over the training region
    h = np.zeros((1, N), np.float32)
    rep_x, rep_h = [], []
    for t in range(n_fit):
        rep_x.append(Xseq[t:t + 1].copy())
        rep_h.append(h.copy())
        h, _ = model([Xseq[t:t + 1].astype(np.float32), h], training=False)
        h = h.numpy()

    def representative():
        for i in range(0, n_fit, 4):
            yield {"x": rep_x[i].astype(np.float32),
                   "h_prev": rep_h[i].astype(np.float32)}

    # float accuracy (sanity)
    pred_f = run_float(model, Xseq, N)
    nrmse_f = common.nrmse(pred_f[test_t], s[test_t + 1])

    # ---- int8 quantize (float I/O so state feedback stays float) ----
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite = conv.convert()
    (OUT / "model_esn_int8.tflite").write_bytes(tflite)

    # int8 host loop with float state feedback (reference kernels)
    nrmse_q, pred_q, h0 = _run_tflite_loop(tflite, Xseq, N, test_t, s)

    # emit model C array + firmware test data (with the warmed-up start state)
    _emit_c_array(tflite, OUT / "model_esn_data.cc", OUT / "model_esn_data.h",
                  "g_esn_model")
    _emit_fw_testdata(tflite, Xseq, N, pred_q, h0)

    result = {
        "arch": f"ESN single-step cell, {N} units (dense W_res {N}x{N})",
        "tflite_bytes": len(tflite),
        "nrmse_float_test": float(nrmse_f),
        "nrmse_int8_test": float(nrmse_q),
        "N": N, "K": K, "M": M,
    }
    (OUT / "esn_tf_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


def _io(interp):
    ins = {d["name"].split(":")[0]: d for d in interp.get_input_details()}
    # name may be "serving_default_x:0" etc; fall back to shape (x is len-K)
    xin = None; hin = None
    for d in interp.get_input_details():
        if d["shape"][-1] == 1:
            xin = d
        else:
            hin = d
    outs = interp.get_output_details()
    yout = None; hout = None
    for d in outs:
        if d["shape"][-1] == 1:
            yout = d
        else:
            hout = d
    return xin, hin, yout, hout


def _run_tflite_loop(tflite, Xseq, N, test_t, s):
    # Use the *reference* kernels (not XNNPACK) so the host loop matches TFLM's
    # reference kernels bit-for-bit — essential for a recurrent model, where
    # per-step kernel differences compound over the feedback loop.
    interp = tf.lite.Interpreter(
        model_content=tflite,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    xin, hin, yout, hout = _io(interp)
    T = Xseq.shape[0]
    h = np.zeros((1, N), np.float32)
    preds = np.zeros(T, np.float64)
    h0 = None
    for t in range(T):
        if t == common.TRAIN_END:
            h0 = h.copy().ravel()           # warmed-up state at the FW start
        interp.set_tensor(xin["index"], Xseq[t:t + 1].astype(np.float32))
        interp.set_tensor(hin["index"], h.astype(np.float32))
        interp.invoke()
        h = interp.get_tensor(hout["index"]).astype(np.float32)
        preds[t] = float(interp.get_tensor(yout["index"]).ravel()[0])
    return common.nrmse(preds[test_t], s[test_t + 1]), preds, h0


def _emit_fw_testdata(tflite, Xseq, N, pred_host, h0):
    """Embed the test input sequence, the warmed-up start state h0, and the
    host int8 reference y (scaled int) so the device replays from the same
    state and can be checked bit-for-bit."""
    start = common.TRAIN_END
    xs = Xseq[start:start + T_FW, 0]
    ys = pred_host[start:start + T_FW]
    yscale = 10000
    yq = np.round(ys * yscale).astype(np.int32)
    h = "\n".join([
        "#ifndef ESN_TEST_DATA_H_",
        "#define ESN_TEST_DATA_H_",
        f"#define ESN_T {T_FW}",
        f"#define ESN_N {N}",
        f"#define ESN_YSCALE {yscale}",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        f"extern const float g_esn_x[{T_FW}];",
        f"extern const float g_esn_h0[{N}];",
        f"extern const int g_esn_yref_scaled[{T_FW}];",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "#endif",
        "",
    ])
    cc = "\n".join([
        '#include "esn_test_data.h"',
        f"const float g_esn_x[{T_FW}] = {{ "
        + ",".join(f"{v:.7g}f" for v in xs) + " };",
        f"const float g_esn_h0[{N}] = {{ "
        + ",".join(f"{v:.9g}f" for v in h0) + " };",
        f"const int g_esn_yref_scaled[{T_FW}] = {{ "
        + ",".join(str(int(v)) for v in yq) + " };",
        "",
    ])
    (FW / "esn_test_data.h").write_text(h)
    (FW / "esn_test_data.cc").write_text(cc)


def _emit_c_array(data, cc_path, h_path, sym):
    body = ",".join(str(b) for b in data)
    cc_path.write_text(
        f'#include "{h_path.name}"\n'
        f"alignas(16) const unsigned char {sym}[] = {{\n{body}\n}};\n"
        f"const unsigned int {sym}_len = {len(data)};\n")
    h_path.write_text(
        "#ifndef ESN_MODEL_DATA_H_\n#define ESN_MODEL_DATA_H_\n"
        '#ifdef __cplusplus\nextern "C" {\n#endif\n'
        f"extern const unsigned char {sym}[];\n"
        f"extern const unsigned int {sym}_len;\n"
        "#ifdef __cplusplus\n}\n#endif\n#endif\n")


if __name__ == "__main__":
    raise SystemExit(main())
