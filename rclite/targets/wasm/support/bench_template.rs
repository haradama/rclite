// Wasmtime benchmark harness for rc_predict.
//
// Embeds X (test input) and Y_REF (host f32-cast reference), runs the
// inference REPEATS+WARMUP times via `std::time::Instant` (WASI clock),
// and prints best/median/mean wall-clock per call plus a parity report
// against the embedded reference.
//
// Output uses keyed `RCLITE_BENCH:` lines so the Python driver in
// `benchmarks/compare_wasm.py` can parse the numbers back without
// depending on float-formatting round-trips.

unsafe extern "C" {
    fn rc_predict(t: i64, x: *const f32, y: *mut f32);
}

const T: usize = @@T@@;
const K: usize = @@K@@;
const M: usize = @@M@@;
const INPUT_LEN: usize = T * K;
const OUTPUT_LEN: usize = T * M;
const REPEATS: usize = @@REPEATS@@;
const WARMUP: usize = @@WARMUP@@;

static X_IN: [f32; INPUT_LEN] = [@@X_VALUES@@];
static Y_REF: [f32; OUTPUT_LEN] = [@@Y_VALUES@@];

fn main() {
    let mut y = vec![0.0_f32; OUTPUT_LEN];

    for _ in 0..WARMUP {
        unsafe { rc_predict(T as i64, X_IN.as_ptr(), y.as_mut_ptr()); }
    }

    let mut samples_ns: Vec<u128> = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t0 = std::time::Instant::now();
        unsafe { rc_predict(T as i64, X_IN.as_ptr(), y.as_mut_ptr()); }
        samples_ns.push(t0.elapsed().as_nanos());
    }

    let mut sse = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for i in 0..OUTPUT_LEN {
        let d = (y[i] as f64) - (Y_REF[i] as f64);
        sse += d * d;
        let a = d.abs();
        if a > max_abs { max_abs = a; }
    }
    let mse = if OUTPUT_LEN > 0 { sse / (OUTPUT_LEN as f64) } else { 0.0 };
    let rmse = mse.sqrt();

    let best_ns: u128 = *samples_ns.iter().min().unwrap_or(&0);
    let worst_ns: u128 = *samples_ns.iter().max().unwrap_or(&0);
    let sum_ns: u128 = samples_ns.iter().sum();
    let mean_ns: u128 = if REPEATS > 0 { sum_ns / (REPEATS as u128) } else { 0 };
    let mut sorted = samples_ns.clone();
    sorted.sort_unstable();
    let median_ns: u128 = if !sorted.is_empty() {
        sorted[sorted.len() / 2]
    } else { 0 };

    println!("RCLITE_BENCH: T={} K={} M={} REPEATS={} WARMUP={}",
             T, K, M, REPEATS, WARMUP);
    println!("RCLITE_BENCH: best_ns={}", best_ns);
    println!("RCLITE_BENCH: median_ns={}", median_ns);
    println!("RCLITE_BENCH: mean_ns={}", mean_ns);
    println!("RCLITE_BENCH: worst_ns={}", worst_ns);
    println!("RCLITE_BENCH: parity_max_abs={:.6e}", max_abs);
    println!("RCLITE_BENCH: parity_rmse={:.6e}", rmse);
    println!("RCLITE_BENCH: y0={:.9e}", y.get(0).copied().unwrap_or(0.0));
    println!("EMULATOR_EXIT");
}
