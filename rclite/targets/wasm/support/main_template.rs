// Wasmtime verification harness for the cross-compiled rc_predict.
//
// Embeds the test input X and the host f64-reference predictions cast to
// f32, runs one inference, and prints a per-step comparison plus
// `EMULATOR_EXIT` so the Python runner can detect success.
//
// Template placeholders (filled in by WasmTarget.compile):
//   @@T@@          -- number of inference timesteps
//   @@K@@          -- input units
//   @@M@@          -- output units
//   @@X_VALUES@@   -- comma-separated input samples (T*K f32 literals)
//   @@Y_VALUES@@   -- comma-separated host-reference predictions (T*M f32)

unsafe extern "C" {
    fn rc_predict(t: i64, x: *const f32, y: *mut f32);
}

const T: usize = @@T@@;
const K: usize = @@K@@;
const M: usize = @@M@@;
const INPUT_LEN: usize = T * K;
const OUTPUT_LEN: usize = T * M;

static X_IN: [f32; INPUT_LEN] = [@@X_VALUES@@];
static Y_REF: [f32; OUTPUT_LEN] = [@@Y_VALUES@@];

fn main() {
    let mut y = vec![0.0_f32; OUTPUT_LEN];
    unsafe { rc_predict(T as i64, X_IN.as_ptr(), y.as_mut_ptr()); }

    let mut sse = 0.0_f32;
    let mut max_abs = 0.0_f32;
    for i in 0..OUTPUT_LEN {
        let d = y[i] - Y_REF[i];
        sse += d * d;
        let a = d.abs();
        if a > max_abs { max_abs = a; }
    }
    let mse = if OUTPUT_LEN > 0 { sse / (OUTPUT_LEN as f32) } else { 0.0 };

    println!("=========================================");
    println!("rc_predict on wasmtime (wasm32-wasip1, f32)");
    println!("=========================================");
    println!("T = {}, K = {}, M = {}", T, K, M);
    for i in 0..T {
        print!("Step {}: Input=[", i);
        for k in 0..K {
            if k > 0 { print!(", "); }
            print!("{:.4}", X_IN[i * K + k]);
        }
        print!("], Ref=[");
        for m in 0..M {
            if m > 0 { print!(", "); }
            print!("{:.4}", Y_REF[i * M + m]);
        }
        print!("], Pred=[");
        for m in 0..M {
            if m > 0 { print!(", "); }
            print!("{:.4}", y[i * M + m]);
        }
        println!("]");
    }
    println!("-----------------------------------------");
    println!("MSE         : {:.6}", mse);
    println!("Max |diff|  : {:.6}", max_abs);
    println!("-----------------------------------------");
    println!("EMULATOR_EXIT");
}
