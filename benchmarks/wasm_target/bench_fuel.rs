// Wasm fuel-measurement harness for rc_predict — float OR integer storage.
//
// Reads the repeat count from the first WASI arg, runs rc_predict that many
// times, then checks the embedded reference. Running the SAME module twice
// with different repeat counts and subtracting wasmtime fuel cancels all
// fixed overhead (WASI startup, the parity loop, argv parsing), leaving
// exactly the fuel of the extra rc_predict calls — a deterministic op-count
// proxy.
//
// Parity tolerance @@EPS@@: 0.5 for integer storage (exact: diff must be 0),
// a small float tolerance otherwise (host f64 reference vs wasm f32 differ by
// rounding; the dense/csr/unroll f32 kernels are mutually bit-exact).
//
// Placeholders filled by bench_wasm.py:
//   @@T@@/@@K@@/@@M@@ — dims; @@STORAGE_T@@ — f32/i8/i16/i32; @@EPS@@ — f64
//   @@X_VALUES@@      — input samples (T*K), raw float or quantized int
//   @@Y_VALUES@@      — reference outputs (T*M)
unsafe extern "C" {
    fn rc_predict(t: i64, x: *const @@STORAGE_T@@, y: *mut @@STORAGE_T@@);
}

const T: usize = @@T@@;
const K: usize = @@K@@;
const M: usize = @@M@@;
const EPS: f64 = @@EPS@@;

static X: [@@STORAGE_T@@; @@T@@ * @@K@@] = [@@X_VALUES@@];
static YREF: [@@STORAGE_T@@; @@T@@ * @@M@@] = [@@Y_VALUES@@];

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

    // Identical work for any reps >= 1, so it cancels in the two-point fuel
    // difference; we read it only for correctness.
    let mut bad = 0usize;
    for i in 0..T * M {
        let d = (y[i] as f64 - YREF[i] as f64).abs();
        if d > EPS { bad += 1; }
    }
    let p = if reps == 0 { "NA" } else if bad == 0 { "OK" } else { "FAIL" };
    println!("reps={} parity={}", reps, p);
}
