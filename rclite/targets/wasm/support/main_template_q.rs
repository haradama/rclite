// Wasmtime harness for the cross-compiled *quantized* rc_predict.
//
// The integer kernel takes storage_t inputs (already at input_scale) and
// writes storage_t outputs (at state_scale). No floating point is touched
// on-device -- tanh is a LUT, so the module pulls in no libm. The embedded
// reference is the host quantized kernel's output, so the comparison is a
// cross-platform integer-determinism check: it must be *exact* (max |diff|
// == 0).
//
// Template placeholders (filled in by rclite.targets.wasm.WasmTarget):
//   @@T@@          -- number of inference timesteps
//   @@K@@          -- input units
//   @@M@@          -- output units
//   @@STORAGE_T@@  -- kernel storage type (i8 / i16 / i32)
//   @@STATE_FRAC@@ -- state Q-format fractional bits (for decoded printing)
//   @@X_VALUES_Q@@ -- comma-separated input samples (at input_scale)
//   @@Y_VALUES_Q@@ -- comma-separated reference outputs (at state_scale)

type Storage = @@STORAGE_T@@;

unsafe extern "C" {
    fn rc_predict(t: i64, x: *const Storage, y: *mut Storage);
}

const T: usize = @@T@@;
const K: usize = @@K@@;
const M: usize = @@M@@;
const STATE_FRAC: u32 = @@STATE_FRAC@@;
const INPUT_LEN: usize = T * K;
const OUTPUT_LEN: usize = T * M;

static X_Q: [Storage; INPUT_LEN] = [@@X_VALUES_Q@@];
static Y_REF_Q: [Storage; OUTPUT_LEN] = [@@Y_VALUES_Q@@];

/// Decode a Q-format fixed-point value to a human-readable f64 (host-side
/// printing only; never used by the kernel).
fn decode(v: i64) -> f64 {
    (v as f64) / ((1_i64 << STATE_FRAC) as f64)
}

fn main() {
    let mut y = vec![0 as Storage; OUTPUT_LEN];
    unsafe { rc_predict(T as i64, X_Q.as_ptr(), y.as_mut_ptr()); }

    let mut max_abs: i64 = 0;
    for i in 0..OUTPUT_LEN {
        let d = (y[i] as i64) - (Y_REF_Q[i] as i64);
        if d.abs() > max_abs { max_abs = d.abs(); }
    }

    println!("=========================================");
    println!("rc_predict on wasmtime (wasm32, quantized {})",
             stringify!(@@STORAGE_T@@));
    println!("=========================================");
    println!("T = {}, K = {}, M = {}, STATE_FRAC = {}", T, K, M, STATE_FRAC);
    for i in 0..T {
        print!("Step {}: Xq=[", i);
        for k in 0..K {
            if k > 0 { print!(", "); }
            print!("{}", X_Q[i * K + k]);
        }
        print!("], Yref=[");
        for m in 0..M {
            if m > 0 { print!(", "); }
            print!("{} ({:.4})", Y_REF_Q[i * M + m], decode(Y_REF_Q[i * M + m] as i64));
        }
        print!("], Y=[");
        for m in 0..M {
            if m > 0 { print!(", "); }
            print!("{} ({:.4})", y[i * M + m], decode(y[i * M + m] as i64));
        }
        println!("]");
    }
    println!("-----------------------------------------");
    println!("Max |Y - Yref| (state-scale units): {}", max_abs);
    println!("                         (decoded) : {:.6}", decode(max_abs));
    println!("-----------------------------------------");
    if max_abs == 0 {
        println!("BIT_EXACT: yes");
    } else {
        println!("BIT_EXACT: no");
    }
    println!("EMULATOR_EXIT");
}
