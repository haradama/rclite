"""Shared result schema + renderers for the per-target performance benches
(Cortex-M0 / AVR / WASM).

Every target reports the SAME columns so the tables line up; a cell that a
target cannot measure is left blank ("" in markdown, "-" in text). Each row
is a dict produced by `row(...)`. The speed metric `ops_per_step` is a
target-native DETERMINISTIC op-count (SysTick ticks / AVR cycles / wasmtime
fuel) — the unit differs per target (stated in the caption), but the
`vs_float` ratio (float-dense / this-row, same N) is unit-free and is the
cross-dtype headline. Size columns: flash_B/ram_B for the MCU targets,
wasm_B for WebAssembly; each target fills only the ones that apply.

DTYPES is the common configuration axis (float + the integer widths);
KERNELS the common W_res strategy axis.
"""
from __future__ import annotations

DTYPES = ["float", "i8", "i16", "i32"]
KERNELS = ["dense", "csr", "unroll"]

# (key, header, kind) — kind drives formatting. A None cell renders as "-"
# (in both markdown and text) so an unmeasured value is never an empty gap.
COLUMNS = [
    ("dtype", "dtype", "str"),
    ("kernel", "kernel", "str"),
    ("flash_B", "Flash B", "int"),
    ("ram_B", "RAM B", "int"),
    ("wasm_B", "wasm B", "int"),
    ("wres_B", "Wres B", "int"),
    ("ops_per_step", "ops/step", "int"),
    ("vs_float", "vs float", "speedup"),
    ("mse", "MSE", "sci"),
    ("parity", "parity", "parity"),
]


def row(*, N, density, nnz, dtype, kernel, ops_per_step=None, parity=None,
        flash_B=None, ram_B=None, wasm_B=None, wres_B=None, mse=None):
    """One measurement. Unmeasured fields stay None → rendered as "-".

    `mse` is the prediction error (mean squared error of the dequantized
    model output vs the ground-truth target, real units) — a function of the
    dtype only (the dense/csr/unroll kernels are bit-exact), so it repeats
    across a dtype's kernel rows.
    """
    return dict(N=N, density=density, nnz=nnz, dtype=dtype, kernel=kernel,
                ops_per_step=ops_per_step, parity=parity, flash_B=flash_B,
                ram_B=ram_B, wasm_B=wasm_B, wres_B=wres_B, mse=mse)


def _add_vs_float(rows):
    """Annotate each row with `vs_float` = float-dense ops / this ops (same N)."""
    base = {}  # N -> float/dense ops_per_step
    for r in rows:
        if r["dtype"] == "float" and r["kernel"] == "dense" \
                and r["ops_per_step"]:
            base[r["N"]] = r["ops_per_step"]
    for r in rows:
        b, o = base.get(r["N"]), r["ops_per_step"]
        r["vs_float"] = (b / o) if (b and o and o > 0) else None


def _cell(r, key, kind, md):
    v = r.get(key)
    if v is None:
        return "-"                       # explicit "not measured" in both
    if kind == "int":
        return str(v)
    if kind == "sci":
        return f"{v:.2e}"
    if kind == "speedup":
        return (f"{v:.2f}×" if md else f"{v:.2f}x")
    if kind == "parity":
        if md:
            return "✅" if v else "❌"
        return "OK" if v else "FAIL"
    return str(v)


def fmt_md(target, rows, *, unit, note=""):
    """GitHub-flavored markdown: one table per N, shared columns, blanks."""
    _add_vs_float(rows)
    out = [f"### {target}", ""]
    cap = (f"`ops/step` = {unit} (deterministic op-count proxy, **not** "
           f"silicon cycles). `vs float` = float-dense ops / row ops at the "
           f"same N. `MSE` = prediction error vs the ground-truth target "
           f"(dequantized output, real units; depends on dtype only). "
           f"A **`-`** cell was **not measured** on this target.")
    if note:
        cap += " " + note
    out += [cap, ""]
    headers = [h for _, h, _ in COLUMNS]
    Ns = sorted({r["N"] for r in rows})
    for N in Ns:
        sub = [r for r in rows if r["N"] == N]
        d = sub[0]["density"]
        nnz = sub[0]["nnz"]
        out.append(f"**N={N}** (density {d:.2f}, nnz {nnz})")
        out.append("")
        out.append("| " + " | ".join(headers) + " |")
        out.append("|" + "|".join(
            (":--" if k == "str" else "--:") if h not in ("parity",)
            else ":--:" for k, h, _ in
            [(c[2], c[1], c[0]) for c in COLUMNS]) + "|")
        for r in sub:
            cells = []
            for key, _, kind in COLUMNS:
                c = _cell(r, key, kind, md=True)
                if key == "kernel" and r["kernel"] == "unroll" and c:
                    c = f"**{c}**"
                cells.append(c)
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def fmt_text(target, rows, *, unit):
    """Plain-text table for terminal output."""
    _add_vs_float(rows)
    widths = {key: max(len(h), 7) for key, h, _ in COLUMNS}
    lines = [f"{target}  [ops/step = {unit}]"]
    Ns = sorted({r["N"] for r in rows})
    for N in Ns:
        sub = [r for r in rows if r["N"] == N]
        lines.append(f"  N={N} density={sub[0]['density']:.2f} "
                     f"nnz={sub[0]['nnz']}")
        hdr = "  " + " ".join(f"{h:>{widths[k]}}" for k, h, _ in COLUMNS)
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for r in sub:
            lines.append("  " + " ".join(
                f"{_cell(r, k, kind, md=False):>{widths[k]}}"
                for k, _, kind in COLUMNS))
    return "\n".join(lines)


def all_parity_ok(rows):
    """True if every row that HAS a parity result passed (None = skipped)."""
    return all(r["parity"] for r in rows if r["parity"] is not None)
