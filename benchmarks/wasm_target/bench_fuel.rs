// Wasm fuel-measurement harness for the symmetric-quantized rc_predict.
//
// Reads the repeat count from the first WASI arg, runs rc_predict that many
// times over the embedded input, then checks bit-exactness against the
// embedded host reference. Running the SAME module twice with different
// repeat counts and subtracting wasmtime fuel cancels all fixed overhead
// (WASI startup, the parity loop, argv parsing), leaving exactly the fuel of
// the extra rc_predict calls — a deterministic op-count proxy.
//
// Template placeholders filled by bench_wasm.py:
//   @@T@@/@@K@@/@@M@@  — dims; @@STORAGE_T@@ — i8/i16/i32
//   @@X_VALUES_Q@@     — input samples at input_scale (T*K)
//   @@Y_VALUES_Q@@     — host-kernel reference outputs at state_scale (T*M)
unsafe extern "C" {
    fn rc_predict(t: i64, x: *const @@STORAGE_T@@, y: *mut @@STORAGE_T@@);
}

const T: usize = @@T@@;
const K: usize = @@K@@;
const M: usize = @@M@@;

static X: [@@STORAGE_T@@; @@T@@ * @@K@@] = [@@X_VALUES_Q@@];
static YREF: [@@STORAGE_T@@; @@T@@ * @@M@@] = [@@Y_VALUES_Q@@];

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut y = [0 as @@STORAGE_T@@; @@T@@ * @@M@@];
    for _ in 0..reps {
        unsafe { rc_predict(T as i64, X.as_ptr(), y.as_mut_ptr()); }
        std::hint::black_box(&y);
    }

    // Parity: identical work for any reps >= 1, so it cancels in the
    // two-point fuel difference; we only read it for correctness.
    let mut mad: i32 = 0;
    for i in 0..T * M {
        let d = (y[i] as i32 - YREF[i] as i32).abs();
        if d > mad { mad = d; }
    }
    let p = if reps == 0 { "NA" } else if mad == 0 { "OK" } else { "FAIL" };
    println!("reps={} parity={}", reps, p);
}
