"""Affine (asymmetric, TFLM-style) LLVM lowering for the RC IDL.

`_AffineLowerer` emits an integer ``rc_predict`` kernel using per-tensor
or per-channel affine quant with TFLM-style requantize;
`emit_quantized_affine_module` is the public entry point. Split out of
the former monolithic ``llvm.py``.
"""

from __future__ import annotations

import numpy as np
from llvmlite import ir
import llvmlite.binding as llvm

from ._llvm_common import (
    _I32,
    _I64,
    _ci,
    _ci32,
    _load1d,
    _load1d_global,
    _load2d_global,
    _loop,
    _loop_strided,
    _pow2_exp,
    _store1d,
)


def emit_quantized_affine_module(
    qmodel, *, passes=None, head=None, vlen=1
) -> ir.Module:
    """Build LLVM IR for the affine integer quantized path (i8 or i16).

    Function signature is identical to the symmetric path:
        void rc_predict(int64_t T, storage_t* X, storage_t* Y);
    With `head="classify"`, Y is `int32_t*` (one class id per step).

    `qmodel` is an `AffineQuantizedModel`; weights and metadata flow
    through `build_ir_from_quantized_affine` into the IR Module, then
    `_AffineLowerer` emits the kernel using TFLM-style requantize.
    """
    from rclite.quant.affine.ir_builder import build_ir_from_quantized_affine

    ir_module = build_ir_from_quantized_affine(qmodel, head=head)
    if passes is None:
        passes = []
    for p in passes:
        ir_module = p(ir_module)
    # Guard the i16 dense matvec vectorization: LLVM fuses it to vpmaddwd, whose
    # i32 pair-sum stays exact iff |W_res_q| <= 32767 (then |W*h| <= 32767*32768
    # < 2^30, pair-sum < 2^31). i8 accumulates in i32 and is always safe. Fall
    # back to scalar when the (rare) i16 quant produces -32768.
    eff_vlen = vlen
    if (
        vlen > 1
        and qmodel.storage_bits == 16
        and qmodel.W_res_q is not None
        and int(np.asarray(qmodel.W_res_q).min()) < -32767
    ):
        eff_vlen = 1
    # Readout W_out_state matvec is safe to vectorize when W_out is i8 (tiny
    # products) or its state block stays in [-32767, 32767] (vpmaddwd guard).
    ro_vec_safe = True
    if vlen > 1 and qmodel.w_out_storage_bits == 16:
        r = qmodel.rc.readout
        off_s = (1 if r.include_bias else 0) + (
            qmodel.K if r.include_input else 0
        )
        ws = np.asarray(qmodel.W_out_q)[:, off_s : off_s + qmodel.N]
        ro_vec_safe = int(ws.min()) >= -32767
    return _AffineLowerer(
        ir_module, vlen=eff_vlen, ro_vec_safe=ro_vec_safe
    ).lower()


class _AffineLowerer:
    """Lower an affine-quantized rclite IR Module to LLVM IR.

    Per-step structure (bit-exact mirror of `AffineQuantizedExecutor`):

        acc_in  = sum_k q_W_in[i,k]  * q_x[k] - zp_u_pre * row_sum_W_in[i]
        acc_res = sum_j q_W_res[i,j] * q_h[j] - zp_state  * row_sum_W_res[i]
        pre[i]  = sat( zp_pre + bias_pre
                       + requantize(acc_in,  M_in_M0,  M_in_n)
                       + requantize(acc_res, M_res_M0, M_res_n) )
        a[i]    = LUT[ sext(pre[i], i32) + lut_offset ]
        h[i]    = sat( zp_state + (h[i] - zp_state)
                       + requantize(a[i] - h[i], leak_M0, leak_n) )

        y[m]    = sat( zp_output
                       + requantize(W_out[m, 0],          M_out_bias_*)   # if include_bias
                       + requantize(W_out·x  − zp·R_in,   M_out_input_*)  # if include_input
                       + requantize(W_out·h  − zp·R_st,   M_out_state_*) )

    Accumulator widths:
        storage_bits == 8  : accum = i32, product = i32
        storage_bits == 16 : accum = i64, product = i64 (matmul over N can
                              overflow i32 for N ≳ 4 with full-range i16)

    The requantize step takes its operand in i32 (after clamping the
    accumulator if needed) and uses `(x*M0 + (1<<(n-1))) >> n`. The
    rounding direction (arithmetic shift on `(prod + bias)`) exactly
    matches what `apply_multiplier_array` does on the Python side, so
    Python and JIT agree bit-for-bit.
    """

    def __init__(self, ir_module, vlen: int = 1, ro_vec_safe: bool = True):
        self.ir_module = ir_module
        md = ir_module.metadata
        if md.get("quantization") != "affine":
            raise ValueError(
                "_AffineLowerer expects metadata['quantization']='affine'"
            )
        # vlen>1 vectorizes the dense W_res AND readout W_out_state matvec
        # reductions. The integer reduction is associative (no mid-loop
        # saturation), so the output is bit-exact with the scalar kernel —
        # quantized SIMD. Default 1 leaves the emitted IR byte-identical (opt-in).
        # `ro_vec_safe` gates the readout vectorization's i16 vpmaddwd guard.
        self.vlen = vlen
        self.ro_vec_safe = ro_vec_safe
        self._init_storage(md)
        self._init_quant_params(md)
        self.K, self.N, self.M = ir_module.K, ir_module.N, ir_module.M
        self._init_function(md)
        self._init_buffers()
        self.t = None

    def _init_storage(self, md) -> None:
        """Storage / accumulator integer types from the quantization width."""
        self.storage_bits = int(md["storage_bits"])
        self.storage_ty = ir.IntType(self.storage_bits)
        if self.storage_bits == 8:
            self.accum_ty = _I32
        elif self.storage_bits == 16:
            self.accum_ty = _I64
        else:
            raise NotImplementedError(
                f"_AffineLowerer only supports storage_bits in {{8, 16}}, "
                f"got {self.storage_bits}"
            )

    def _init_quant_params(self, md) -> None:
        """Parse affine quantization parameters out of metadata: zero points,
        requantize multipliers, LUT strategy, topology, integer preprocess."""
        # Zero points (Python ints)
        self.zp_input = int(md["zp_input"])
        self.zp_u_pre = int(md["zp_u_pre"])
        self.zp_state = int(md["zp_state"])
        self.zp_pre = int(md["zp_pre"])
        self.zp_output = int(md["zp_output"])

        self.lut_offset = int(md["lut_offset"])
        self.bias_pre = int(md["bias_pre"])

        # LUT strategy and per-strategy precomputed.
        self.lut_kind = md.get("lut_kind", "direct")
        if self.lut_kind == "linear_interp":
            self.lut_n_entries = int(md["lut_n_entries"])
            self.lut_interp_frac_bits = int(md["lut_interp_frac_bits"])
            self.lut_idx_M0 = int(md["lut_idx_M0"])
            self.lut_idx_n = int(md["lut_idx_n"])
        elif self.lut_kind == "polynomial":
            self.poly_qf_bits = int(md["poly_qf_bits"])
            self.poly_degree = int(md.get("poly_degree", 5))
            self.poly_x_M0 = int(md["poly_x_M0"])
            self.poly_x_n = int(md["poly_x_n"])
            self.poly_back_M0 = int(md["poly_back_M0"])
            self.poly_back_n = int(md["poly_back_n"])
            self.poly_clip_qf = int(md["poly_clip_qf"])
            self.poly_one_qf = int(md["poly_one_qf"])
            self.poly_a1_qf = int(md["poly_a1_qf"])
            self.poly_a3_qf = int(md["poly_a3_qf"])
            self.poly_a5_qf = int(md["poly_a5_qf"])

        # Requantize multipliers (M0, n)
        self.M_in_M0, self.M_in_n = int(md["M_in_M0"]), int(md["M_in_n"])
        self.M_res_M0, self.M_res_n = int(md["M_res_M0"]), int(md["M_res_n"])
        self.per_channel_res = bool(md.get("per_channel_res", False))
        self.per_channel_out = bool(md.get("per_channel_out", False))
        self.leak_M0, self.leak_n = int(md["leak_M0"]), int(md["leak_n"])
        self.M_out_bias_M0 = int(md["M_out_bias_M0"])
        self.M_out_bias_n = int(md["M_out_bias_n"])
        self.M_out_input_M0 = int(md["M_out_input_M0"])
        self.M_out_input_n = int(md["M_out_input_n"])
        self.M_out_state_M0 = int(md["M_out_state_M0"])
        self.M_out_state_n = int(md["M_out_state_n"])

        self.include_bias = bool(md["include_bias"])
        self.include_input = bool(md["include_input"])

        # Topology specialisation (SCR/DLR/DLRB skip the dense W_res matmul).
        self.structured = bool(md.get("structured", False))
        self.topology_name = md.get("topology", "ESN_STANDARD")
        self.chain_weight_q = int(md.get("chain_weight_q", 0))
        self.chain_feedback_q = int(md.get("chain_feedback_q", 0))

        # Integer input preprocess (active iff input_offset != 0 or
        # input_scaling != 1). When active, the kernel computes u_pre into
        # a scratch buffer that the W_in matmul reads; otherwise the matmul
        # reads X directly.
        self.has_int_preprocess = bool(md.get("has_integer_preprocess", False))
        self.pre_M0 = int(md.get("pre_M0", 0))
        self.pre_n = int(md.get("pre_n", 0))
        self.pre_const = int(md.get("pre_const", 0))

    def _init_function(self, md) -> None:
        """Create the LLVM module, detect the output head (argmax/softmax/
        mean-pool), emit the weight globals and declare `rc_predict`."""
        from rclite.ir.ops import (
            Argmax,
            Softmax,
            AccumulateState,
        )

        self.module = ir.Module(
            name=f"rc_affine_jit_i{self.storage_bits}_{id(self.ir_module)}",
        )
        self.module.triple = llvm.get_default_triple()

        # Classification head: argmax emits an int32 class id per step;
        # softmax emits M probabilities (storage type, Q.sm_prob_frac).
        # Both route the readout through a logits scratch.
        flat = list(self._flatten_ops())
        self.out_int = any(isinstance(op, Argmax) for op in flat)
        self.has_softmax = any(isinstance(op, Softmax) for op in flat)
        self.has_head = self.out_int or self.has_softmax
        # MEAN time-pooling needs a running i64 state-sum buffer.
        self.needs_state_sum = any(
            isinstance(op, AccumulateState) and op.mode == "mean"
            for op in flat
        )
        out_ty = _I32 if self.out_int else self.storage_ty
        if self.has_softmax:
            self.sm_dmin_q = int(md["sm_dmin_q"])
            self.sm_n = int(md["sm_n"])
            self.sm_idx_frac = int(md["sm_idx_frac"])
            self.sm_prob_frac = int(md["sm_prob_frac"])

        # Emit all globals (storage-typed weights + i32 precomputed row sums).
        self.globals = {}
        for name, arr in self.ir_module.weights.items():
            self.globals[name] = self._emit_global(name, arr)

        # void rc_predict(i64 T, storage_t* X, {storage_t|i32}* Y)
        fnty = ir.FunctionType(
            ir.VoidType(),
            [_I64, self.storage_ty.as_pointer(), out_ty.as_pointer()],
        )
        self.fn = ir.Function(self.module, fnty, name="rc_predict")
        self.T_arg, self.X_arg, self.Y_arg = self.fn.args
        self.T_arg.name = "T"
        self.X_arg.name = "X"
        self.Y_arg.name = "Y"

    def _init_buffers(self) -> None:
        """Open the entry block and allocate the per-run scratch buffers
        (state, pre-activation, logits/exp, u_pre, state-sum), then zero-init
        the reservoir state to `zp_state`."""
        entry = self.fn.append_basic_block("entry")
        self.b = ir.IRBuilder(entry)

        # Buffers
        self.h_buf = self.b.alloca(
            self.storage_ty,
            size=_ci(self.N),
            name="h",
        )
        self.pre_buf = self.b.alloca(
            self.storage_ty,
            size=_ci(self.N),
            name="pre",
        )
        self.logits = (
            self.b.alloca(
                self.storage_ty, size=_ci(max(self.M, 1)), name="logits"
            )
            if self.has_head
            else None
        )
        self.exp_scratch = (
            self.b.alloca(_I32, size=_ci(max(self.M, 1)), name="exp_q")
            if self.has_softmax
            else None
        )
        # u_pre scratch only allocated when integer preprocess is in play —
        # otherwise the matmul reads X directly.
        if self.has_int_preprocess:
            self.u_pre_buf = self.b.alloca(
                self.storage_ty,
                size=_ci(max(self.K, 1)),
                name="u_pre",
            )
        else:
            self.u_pre_buf = None

        # Running state-sum buffer (i64) for MEAN time-pooling.
        if self.needs_state_sum:
            self.h_sum = self.b.alloca(_I64, size=_ci(self.N), name="h_sum")
            with _loop(self.b, _ci(self.N), "sinit") as i:
                _store1d(self.b, self.h_sum, i, self._ci64(0))
        else:
            self.h_sum = None

        # Initialize state to zp_state
        with _loop(self.b, _ci(self.N), "init") as i:
            _store1d(self.b, self.h_buf, i, self._cs(self.zp_state))

    # ------------------------------------------------------------------
    # constants

    def _cs(self, v: int) -> ir.Constant:
        """Constant in storage_ty (i8 or i16)."""
        return ir.Constant(self.storage_ty, int(v))

    def _ca(self, v: int) -> ir.Constant:
        """Constant in accum_ty (i32 or i64)."""
        return ir.Constant(self.accum_ty, int(v))

    def _ci32(self, v: int) -> ir.Constant:
        return ir.Constant(_I32, int(v))

    def _ci64(self, v: int) -> ir.Constant:
        return ir.Constant(_I64, int(v))

    # ------------------------------------------------------------------
    # global emission

    def _emit_global(self, name, arr):
        flat = np.asarray(arr).reshape(-1)
        if flat.dtype == np.int8:
            elem_ty = ir.IntType(8)
        elif flat.dtype == np.int16:
            elem_ty = ir.IntType(16)
        elif flat.dtype == np.int32:
            elem_ty = ir.IntType(32)
        else:
            raise ValueError(
                f"_AffineLowerer global {name!r}: unsupported dtype "
                f"{flat.dtype}"
            )
        arr_ty = ir.ArrayType(elem_ty, flat.size)
        g = ir.GlobalVariable(self.module, arr_ty, name=name)
        g.linkage = "internal"
        g.global_constant = True
        g.initializer = ir.Constant(
            arr_ty,
            [ir.Constant(elem_ty, int(v)) for v in flat],
        )
        return g

    # ------------------------------------------------------------------
    # core helpers

    def _widen_to_accum(self, v):
        """Sign-extend an integer SSA value up to `accum_ty`.

        Row-sum globals are stored at their minimal signed width to save
        Flash; their values always fit `accum_ty`, so a plain `sext` (or a
        pass-through when already that width) restores the exact value.
        """
        if v.type.width < self.accum_ty.width:
            return self.b.sext(v, self.accum_ty)
        return v

    def _clamp_to_i32(self, val_ty):
        """If accumulator is i64, clamp to i32 range and truncate; else passthrough."""
        if self.accum_ty == _I32:
            return val_ty
        return self._clamp_i64_to_i32(val_ty)

    def _clamp_i64_to_i32(self, val_i64):
        """Clamp an i64 value to the signed i32 range and truncate to i32."""
        lo = self._ci64(-(1 << 31))
        hi = self._ci64((1 << 31) - 1)
        clipped_lo = self.b.select(
            self.b.icmp_signed("<", val_i64, lo),
            lo,
            val_i64,
        )
        clipped = self.b.select(
            self.b.icmp_signed(">", clipped_lo, hi),
            hi,
            clipped_lo,
        )
        return self.b.trunc(clipped, _I32)

    def _emit_requantize_i32(self, acc_i32, M0: int, n: int):
        """Compute `(acc * M0 + (1<<(n-1))) >> n` with i64 product. Returns i32.

        Matches `apply_multiplier_array` in the Python ref bit-for-bit.
        """
        if M0 == 0:
            return self._ci32(0)
        acc_64 = self.b.sext(acc_i32, _I64)
        prod = self.b.mul(acc_64, self._ci64(M0))
        if n > 0:
            prod = self.b.add(prod, self._ci64(1 << (n - 1)))
        shr = self.b.ashr(prod, self._ci64(n))
        return self.b.trunc(shr, _I32)

    def _emit_requantize_i32_dynamic(self, acc_i32, M0_i32, n_i32):
        """Runtime-shift requantize `(acc*M0 + (1<<(n-1)))>>n`, M0/n as SSA i32.

        Per-channel variant: M0 and n are loaded per reservoir row, so the
        shift amount is dynamic. Matches `apply_multiplier_perrow` bit-for-bit
        (n==0 → no rounding bias, ashr by 0 is identity).
        """
        b = self.b
        acc_64 = b.sext(acc_i32, _I64)
        m0_64 = b.sext(M0_i32, _I64)
        n_64 = b.sext(n_i32, _I64)
        prod = b.mul(acc_64, m0_64)
        nz = b.icmp_signed(">", n_64, self._ci64(0))
        safe_sh = b.select(nz, b.sub(n_64, self._ci64(1)), self._ci64(0))
        half = b.select(nz, b.shl(self._ci64(1), safe_sh), self._ci64(0))
        prod = b.add(prod, half)
        shr = b.ashr(prod, n_64)
        return b.trunc(shr, _I32)

    def _emit_saturate_to_storage(self, val_i32):
        """Clamp i32 to signed storage range, then truncate to storage_ty."""
        lo = self._ci32(-(1 << (self.storage_bits - 1)))
        hi = self._ci32((1 << (self.storage_bits - 1)) - 1)
        clipped_lo = self.b.select(
            self.b.icmp_signed("<", val_i32, lo),
            lo,
            val_i32,
        )
        clipped = self.b.select(
            self.b.icmp_signed(">", clipped_lo, hi),
            hi,
            clipped_lo,
        )
        return self.b.trunc(clipped, self.storage_ty)

    # ------------------------------------------------------------------
    # dispatcher

    def lower(self) -> ir.Module:
        for op in self.ir_module.ops:
            self._lower(op)
        self.b.ret_void()
        return self.module

    def _flatten_ops(self):
        from rclite.ir.ops import TimeLoop

        for op in self.ir_module.ops:
            yield op
            if isinstance(op, TimeLoop):
                yield from op.body

    def _lower(self, op):
        from rclite.ir.ops import (
            TimeLoop,
            PreprocessInput,
            ReservoirStep,
            BuildPhi,
            ReadoutLinear,
            Argmax,
            Softmax,
            AccumulateState,
            FinalizeAggregate,
        )

        if isinstance(op, TimeLoop):
            return self._lower_time_loop(op)
        if isinstance(op, PreprocessInput):
            return self._lower_preprocess_affine(op)
        if isinstance(op, ReservoirStep):
            return self._lower_reservoir_step(op)
        if isinstance(op, BuildPhi):
            # Affine readout pulls X and h directly — no phi buffer needed.
            return
        if isinstance(op, ReadoutLinear):
            return self._lower_readout_linear(op)
        if isinstance(op, AccumulateState):
            return self._lower_accumulate_state(op)
        if isinstance(op, FinalizeAggregate):
            return self._lower_finalize_aggregate(op)
        if isinstance(op, Argmax):
            return self._lower_argmax(op)
        if isinstance(op, Softmax):
            return self._lower_softmax(op)
        raise NotImplementedError(
            f"{type(op).__name__} not supported in the affine path"
        )

    # ------------------------------------------------------------------
    # sequence-to-label time pooling (mirrors AffineQuantizedExecutor)

    def _washout_clamped(self, washout):
        """Return min(washout, T-1) clamped at >= 0 as an i64 SSA value."""
        b = self.b
        w_const = self._ci64(washout)
        t_minus1 = b.sub(self.T_arg, self._ci64(1))
        w = b.select(
            b.icmp_signed("<", w_const, self.T_arg), w_const, t_minus1
        )
        return b.select(b.icmp_signed("<", w, self._ci64(0)), self._ci64(0), w)

    def _lower_accumulate_state(self, op):
        """mode='mean': h_sum[i] += q_h[i] for t >= washout. 'last': no-op."""
        if op.mode == "last":
            return
        b = self.b
        w = self._washout_clamped(op.washout)
        in_window = b.icmp_signed(">=", self.t, w)
        with _loop(b, _ci(op.N), "acc_h") as i:
            s = _load1d(b, self.h_sum, i)  # i64
            h_i = b.sext(_load1d(b, self.h_buf, i), _I64)
            add = b.select(in_window, h_i, self._ci64(0))
            _store1d(b, self.h_sum, i, b.add(s, add))

    def _lower_finalize_aggregate(self, op):
        """Write the pooled state back into h_buf, then point output at row 0.

        mode='mean' divides the running sum by L = T - washout, rounding half
        away from zero (bit-exact with `AffineQuantizedExecutor._round_div`).
        mode='last' leaves the final state in place.
        """
        if op.mode == "mean":
            b = self.b
            w = self._washout_clamped(op.washout)
            L = b.sub(self.T_arg, w)  # i64, >= 1
            with _loop(b, _ci(op.N), "fin_h") as i:
                s = _load1d(b, self.h_sum, i)  # i64
                q = self._emit_round_div_i64(s, L)
                _store1d(
                    b,
                    self.h_buf,
                    i,
                    self._emit_saturate_to_storage(self._clamp_i64_to_i32(q)),
                )
        # The pooled sequence produces a single output row.
        self.t = _ci(0)

    def _emit_round_div_i64(self, s, L):
        """Round-half-away-from-zero integer division s/L (L>0), in i64."""
        b = self.b
        half = b.ashr(L, self._ci64(1))  # floor(L/2), L>0
        is_neg = b.icmp_signed("<", s, self._ci64(0))
        pos = b.sdiv(b.add(s, half), L)  # sdiv truncates toward 0
        neg_s = b.sub(self._ci64(0), s)
        neg = b.sub(self._ci64(0), b.sdiv(b.add(neg_s, half), L))
        return b.select(is_neg, neg, pos)

    def _lower_argmax(self, op):
        """class_id = argmax_m logits[m] over the monotone quantized scores."""
        b = self.b
        best_v = b.alloca(self.storage_ty, name="best_v")
        best_i = b.alloca(_I64, name="best_i")
        b.store(_load1d(b, self.logits, _ci(0)), best_v)
        b.store(_ci(0), best_i)
        with _loop(b, _ci(op.M), "am") as m:
            v = _load1d(b, self.logits, m)
            is_gt = b.icmp_signed(">", v, b.load(best_v))
            b.store(b.select(is_gt, v, b.load(best_v)), best_v)
            b.store(b.select(is_gt, m, b.load(best_i)), best_i)
        _store1d(b, self.Y_arg, self.t, b.trunc(b.load(best_i), _I32))

    def _lower_softmax(self, op):
        """Fixed-point softmax (exp LUT), bit-exact with softmax_q.

        Identical integer algorithm to the symmetric path; operates on the
        quantized logits scratch and writes Q.sm_prob_frac probabilities.
        """
        b = self.b
        g_lut = self.globals["sm_lut"]
        n = self.sm_n
        idxf = self.sm_idx_frac
        dmin = self.sm_dmin_q
        pf = self.sm_prob_frac
        M = op.M
        qmax = (1 << (self.storage_bits - 1)) - 1

        mx = b.alloca(_I32, name="sm_max")
        b.store(b.sext(_load1d(b, self.logits, _ci(0)), _I32), mx)
        with _loop(b, _ci(M), "smx") as m:
            v = b.sext(_load1d(b, self.logits, m), _I32)
            b.store(
                b.select(b.icmp_signed(">", v, b.load(mx)), v, b.load(mx)), mx
            )

        sum_acc = b.alloca(_I64, name="sm_sum")
        b.store(self._ci64(0), sum_acc)
        with _loop(b, _ci(M), "sme") as m:
            v = b.sext(_load1d(b, self.logits, m), _I32)
            d = b.sub(v, b.load(mx))
            d = b.select(
                b.icmp_signed("<", d, self._ci32(dmin)), self._ci32(dmin), d
            )
            num = b.sub(d, self._ci32(dmin))
            num64 = b.sext(num, _I64)
            posn = b.shl(b.mul(num64, self._ci64(n - 1)), self._ci64(idxf))
            pos = b.sdiv(posn, self._ci64(-dmin))
            i0 = b.ashr(pos, self._ci64(idxf))
            i0 = b.select(
                b.icmp_signed("<", i0, self._ci64(0)), self._ci64(0), i0
            )
            i0 = b.select(
                b.icmp_signed(">", i0, self._ci64(n - 2)),
                self._ci64(n - 2),
                i0,
            )
            frac = b.sub(pos, b.shl(i0, self._ci64(idxf)))
            y0 = b.sext(_load1d_global(b, g_lut, i0), _I64)
            y1 = b.sext(
                _load1d_global(b, g_lut, b.add(i0, self._ci64(1))), _I64
            )
            e = b.add(y0, b.ashr(b.mul(b.sub(y1, y0), frac), self._ci64(idxf)))
            _store1d(b, self.exp_scratch, m, b.trunc(e, _I32))
            b.store(b.add(b.load(sum_acc), e), sum_acc)

        s = b.load(sum_acc)
        with _loop(b, _ci(M), "smn") as m:
            e = b.sext(_load1d(b, self.exp_scratch, m), _I64)
            p = b.sdiv(b.shl(e, self._ci64(pf)), s)
            p = b.select(
                b.icmp_signed(">", p, self._ci64(qmax)), self._ci64(qmax), p
            )
            tM = b.mul(self.t, _ci(M))
            _store1d(b, self.Y_arg, b.add(tM, m), b.trunc(p, self.storage_ty))

    def _lower_preprocess_affine(self, op):
        """Integer preprocess: u_pre[k] = sat(pre_const + apply_mult(q_x − zp_x)).

        Mirrors `AffineQuantizedExecutor._quantize_u_pre` step for step.
        """
        K = op.K
        if K == 0 or not self.has_int_preprocess:
            return
        t = self.t
        tK = self.b.mul(t, _ci(K))
        with _loop(self.b, _ci(K), "kpre_aff") as k:
            x_q = _load1d(self.b, self.X_arg, self.b.add(tK, k))
            centered = self.b.sub(
                self.b.sext(x_q, _I32),
                self._ci32(self.zp_input),
            )
            delta = self._emit_requantize_i32(
                centered,
                self.pre_M0,
                self.pre_n,
            )
            total = self.b.add(delta, self._ci32(self.pre_const))
            _store1d(
                self.b,
                self.u_pre_buf,
                k,
                self._emit_saturate_to_storage(total),
            )

    def _lower_time_loop(self, op):
        with _loop(self.b, self.T_arg, "t") as t:
            self.t = t
            for body_op in op.body:
                self._lower(body_op)
        self.t = None

    # ------------------------------------------------------------------
    # reservoir step

    def _lower_reservoir_step(self, op):
        g_Win = self.globals["W_in"]
        g_rs_in = self.globals["row_sum_W_in"]
        # W_res / row_sum_W_res only exist for non-structured (dense) topologies.
        g_Wres = self.globals.get("W_res")
        g_rs_res = self.globals.get("row_sum_W_res")
        K, N = op.K, op.N
        t = self.t

        # ---- Pre-act loop ----
        spec = op.res_sparse
        if spec is not None and spec.kind == "unroll":
            # Per-row nonzero sets differ → unroll the outer i-loop.
            for i in range(N):
                self._emit_affine_row(
                    op, _ci(i), g_Win, g_rs_in, g_Wres, g_rs_res, spec, i_py=i
                )
        else:
            with _loop(self.b, _ci(N), "ipre") as i:
                self._emit_affine_row(
                    op, i, g_Win, g_rs_in, g_Wres, g_rs_res, spec, i_py=None
                )

        # ---- Activation + leaky integration ----
        with _loop(self.b, _ci(N), "iact") as i:
            p = _load1d(self.b, self.pre_buf, i)  # storage_ty
            a = self._emit_activation(p)  # storage_ty

            h_old = _load1d(self.b, self.h_buf, i)
            h_c = self.b.sub(
                self.b.sext(h_old, _I32), self._ci32(self.zp_state)
            )
            a_c = self.b.sub(self.b.sext(a, _I32), self._ci32(self.zp_state))
            diff = self.b.sub(a_c, h_c)
            delta = self._emit_requantize_i32(diff, self.leak_M0, self.leak_n)
            new_h_c = self.b.add(h_c, delta)
            new_h_total = self.b.add(new_h_c, self._ci32(self.zp_state))
            new_h_q = self._emit_saturate_to_storage(new_h_total)
            _store1d(self.b, self.h_buf, i, new_h_q)

    def _const_mul_accum(self, wv: int, h):
        """wv * sext(h) in accum_ty, folding the multiply when wv==+-2**k.

        `mul(2**k, sext(h))` equals `shl(sext(h), k)` bit-for-bit in accum_ty
        (no overflow: accum_ty is wider than the storage state), and a
        negative power negates the shifted value -- bit-identical to the
        baked `mul`. Falls back to the multiply otherwise.
        """
        b = self.b
        k = _pow2_exp(wv)
        h_p = b.sext(h, self.accum_ty)
        if k is None:
            return b.mul(self._ca(int(wv)), h_p)
        if k > 0:
            h_p = b.shl(h_p, ir.Constant(self.accum_ty, k))
        if wv < 0:
            h_p = b.sub(self._ca(0), h_p)
        return h_p

    def _emit_affine_row(
        self, op, i, g_Win, g_rs_in, g_Wres, g_rs_res, spec, i_py
    ):
        """Emit pre[row i] for the affine kernel (one body of the ipre loop).

        `i` is an SSA index (a constant when unrolling). For the unrolled
        sparse kernel `i_py` is the Python row index and the recurrent
        accumulation uses the baked nonzeros in `spec.rows[i_py]`; the
        affine zero-point correction `- zp_state * row_sum_W_res[i]` and the
        requantize are unchanged (row_sum_W_res is preserved by the pass).
        """
        rq_in = self._emit_preact_input(op, i, g_Win, g_rs_in)
        rq_res = self._emit_preact_reservoir(
            op, i, g_Wres, g_rs_res, spec, i_py
        )
        pre_total = self.b.add(
            self.b.add(self._ci32(self.zp_pre + self.bias_pre), rq_in), rq_res
        )
        pre_q = self._emit_saturate_to_storage(pre_total)
        _store1d(self.b, self.pre_buf, i, pre_q)

    def _emit_preact_input(self, op, i, g_Win, g_rs_in):
        """W_in·x for row i, minus the `zp_u_pre · row_sum_W_in[i]` affine
        correction, requantized to i32 (the input pre-activation term)."""
        b, K, t = self.b, op.K, self.t
        acc_in_var = b.alloca(self.accum_ty, name="acc_in")
        b.store(self._ca(0), acc_in_var)
        with _loop(b, _ci(K), "kin") as k:
            w = _load2d_global(b, g_Win, K, i, k)
            if self.has_int_preprocess:
                x = _load1d(b, self.u_pre_buf, k)
            else:
                x = _load1d(b, self.X_arg, b.add(b.mul(t, _ci(K)), k))
            prod = b.mul(b.sext(w, self.accum_ty), b.sext(x, self.accum_ty))
            b.store(b.add(b.load(acc_in_var), prod), acc_in_var)
        rs_in = self._widen_to_accum(_load1d_global(b, g_rs_in, i))
        acc_in_final = b.sub(
            b.load(acc_in_var), b.mul(self._ca(self.zp_u_pre), rs_in)
        )
        return self._emit_requantize_i32(
            self._clamp_to_i32(acc_in_final), self.M_in_M0, self.M_in_n
        )

    def _emit_preact_reservoir(self, op, i, g_Wres, g_rs_res, spec, i_py):
        """W_res·h for row i — structured chain / unrolled-sparse / CSR /
        dense — minus the `zp_state · row_sum_W_res[i]` affine correction,
        requantized to i32 (per-channel `(M0[i], n[i])` when enabled)."""
        b, N = self.b, op.N
        if self.structured:
            acc_res_i32 = self._emit_chain_contribution(i, N)
        else:
            acc_res_var = b.alloca(self.accum_ty, name="acc_res")
            b.store(self._ca(0), acc_res_var)
            if i_py is not None:  # unrolled sparse
                for j, wv in spec.rows[i_py]:
                    h = _load1d(b, self.h_buf, _ci(j))
                    prod = self._const_mul_accum(int(wv), h)
                    b.store(b.add(b.load(acc_res_var), prod), acc_res_var)
            elif spec is not None:  # CSR
                self._emit_affine_res_csr(spec, acc_res_var, i)
            elif self.vlen > 1:  # dense, vectorized
                self._emit_dense_res_vec(g_Wres, i, acc_res_var)
            else:  # dense, scalar
                with _loop(b, _ci(N), "jres") as j:
                    w = _load2d_global(b, g_Wres, N, i, j)
                    h = _load1d(b, self.h_buf, j)
                    prod = b.mul(
                        b.sext(w, self.accum_ty), b.sext(h, self.accum_ty)
                    )
                    b.store(b.add(b.load(acc_res_var), prod), acc_res_var)
            rs_res = self._widen_to_accum(_load1d_global(b, g_rs_res, i))
            acc_res_final = b.sub(
                b.load(acc_res_var), b.mul(self._ca(self.zp_state), rs_res)
            )
            acc_res_i32 = self._clamp_to_i32(acc_res_final)
        if self.per_channel_res and not self.structured:
            # per-row (M0[i], n[i]) loaded from i32 globals → dynamic shift.
            m0_i = _load1d_global(b, self.globals["M_res_M0"], i)
            n_i = _load1d_global(b, self.globals["M_res_n"], i)
            return self._emit_requantize_i32_dynamic(acc_res_i32, m0_i, n_i)
        return self._emit_requantize_i32(
            acc_res_i32, self.M_res_M0, self.M_res_n
        )

    def _vec_reduce_add(self, vec, elem_ty, vlen):
        """`llvm.vector.reduce.add` -> scalar. Integer add is associative, so the
        lane reduction equals the linear scalar sum bit-for-bit."""
        name = f"llvm.vector.reduce.add.v{vlen}i{elem_ty.width}"
        fn = self.module.globals.get(name)
        if fn is None:
            fn = ir.Function(
                self.module,
                ir.FunctionType(elem_ty, [ir.VectorType(elem_ty, vlen)]),
                name,
            )
        return self.b.call(fn, [vec])

    def _emit_int_matvec_vec(
        self, w_base, h_base, N, acc_var, acc_ty, w_sty, h_sty, tag
    ):
        """`acc_var += sum_j sext(W[base+j]) * sext(h[j])`, vectorized.

        Multiplies in i32 (an i8/i16 product is <= 2^30, fits i32 exactly) and
        accumulates in a `<vlen x acc_ty>` lane vector, collapsed with
        `vector.reduce.add` plus a scalar tail. Bit-exact with the scalar loop
        (the integer sum is associative). `w_base`/`h_base` are element pointers
        (W and h may have different storage types — the readout's mixed-precision
        W_out[iwob]*h[isb]); loads are element-aligned. On i16, LLVM fuses the i32
        widening-multiply into `vpmaddwd`."""
        b, vlen = self.b, self.vlen
        Nv = (N // vlen) * vlen
        wvt, hvt = ir.VectorType(w_sty, vlen), ir.VectorType(h_sty, vlen)
        i32vt, accvt = ir.VectorType(_I32, vlen), ir.VectorType(acc_ty, vlen)
        if Nv > 0:
            vacc = b.alloca(accvt, name="vacc")
            b.store(ir.Constant(accvt, None), vacc)
            with _loop_strided(b, _ci(0), _ci(Nv), _ci(vlen), "v" + tag) as j:
                wp = b.bitcast(b.gep(w_base, [j]), wvt.as_pointer())
                hp = b.bitcast(b.gep(h_base, [j]), hvt.as_pointer())
                wv = b.load(wp, align=w_sty.width // 8)
                hv = b.load(hp, align=h_sty.width // 8)
                prod = b.mul(b.sext(wv, i32vt), b.sext(hv, i32vt))
                if acc_ty != _I32:
                    prod = b.sext(prod, accvt)
                b.store(b.add(b.load(vacc), prod), vacc)
            red = self._vec_reduce_add(b.load(vacc), acc_ty, vlen)
            b.store(b.add(b.load(acc_var), red), acc_var)
        if Nv < N:
            with _loop_strided(b, _ci(Nv), _ci(N), _ci(1), "t" + tag) as j:
                w = b.load(b.gep(w_base, [j]))
                h = b.load(b.gep(h_base, [j]))
                prod = b.mul(b.sext(w, acc_ty), b.sext(h, acc_ty))
                b.store(b.add(b.load(acc_var), prod), acc_var)

    def _emit_dense_res_vec(self, g_Wres, i, acc_res_var):
        """Vectorized dense W_res·h for row i (W and h are both the state
        storage type). &W_res[i*N] is the row base pointer."""
        w_base = self.b.gep(g_Wres, [_ci32(0), self.b.mul(i, _ci(self.N))])
        self._emit_int_matvec_vec(
            w_base,
            self.h_buf,
            self.N,
            acc_res_var,
            self.accum_ty,
            self.storage_ty,
            self.storage_ty,
            "jres",
        )

    def _emit_affine_res_csr(self, spec, acc_res_var, i):
        """Accumulate W_res·h over row i's nonzeros (CSR) into acc_res_var."""
        b = self.b
        g_val = self.globals[spec.val_name]
        g_col = self.globals[spec.col_name]
        g_rowptr = self.globals[spec.rowptr_name]
        start = b.sext(_load1d_global(b, g_rowptr, i), _I64)
        end = b.sext(_load1d_global(b, g_rowptr, b.add(i, _ci(1))), _I64)
        with _loop_strided(b, start, end, _ci(1), "csr") as p:
            j = b.sext(_load1d_global(b, g_col, p), _I64)
            w = _load1d_global(b, g_val, p)
            h = _load1d(b, self.h_buf, j)
            prod = b.mul(b.sext(w, self.accum_ty), b.sext(h, self.accum_ty))
            b.store(b.add(b.load(acc_res_var), prod), acc_res_var)

    # ------------------------------------------------------------------
    # structured-topology W_res contribution (SCR / DLR / DLRB)

    def _emit_chain_contribution(self, i, N):
        """Return the i32 acc_res for row `i` under a structured topology.

        Algebraic identity (since q_W_res is symmetric, zp_W_res = 0):
            sum_j q_W[i,j]·q_h[j]  −  zp_state·row_sum_W[i]
          =  cw_q · (q_h[prev] − zp_state)              for SCR/DLR(i>0)
          +  cf_q · (q_h[next] − zp_state)              for DLRB extra edge

        Each chain entry is a single i8/i16 weight, so the product fits
        in i32 without a wider accumulator.
        """
        b = self.b
        cw = self._ci32(self.chain_weight_q)
        cf = self._ci32(self.chain_feedback_q)
        zp_state_const = self._ci32(self.zp_state)
        zero32 = self._ci32(0)

        def _h_centered(idx_i64):
            h_val = _load1d(b, self.h_buf, idx_i64)
            return b.sub(b.sext(h_val, _I32), zp_state_const)

        topo = self.topology_name
        if topo == "SCR":
            # prev_idx = (i==0 ? N-1 : i-1)
            is_zero = b.icmp_signed("==", i, _ci(0))
            i_prev = b.select(is_zero, _ci(N - 1), b.sub(i, _ci(1)))
            return b.mul(cw, _h_centered(i_prev))
        if topo == "DLR":
            # only contribute for i > 0
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_safe = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            prod = b.mul(cw, _h_centered(i_safe))
            return b.select(is_pos, prod, zero32)
        if topo == "DLRB":
            # backward chain: chain_weight * h[i-1] for i > 0
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_back = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            back_prod = b.mul(cw, _h_centered(i_back))
            contrib_back = b.select(is_pos, back_prod, zero32)
            # forward chain: chain_feedback * h[i+1] for i < N-1
            is_lt = b.icmp_signed("<", i, _ci(N - 1))
            i_fwd = b.select(is_lt, b.add(i, _ci(1)), _ci(N - 1))
            fwd_prod = b.mul(cf, _h_centered(i_fwd))
            contrib_fwd = b.select(is_lt, fwd_prod, zero32)
            return b.add(contrib_back, contrib_fwd)
        raise ValueError(
            f"_emit_chain_contribution: unsupported structured topology "
            f"{topo!r}"
        )

    # ------------------------------------------------------------------
    # activation — dispatch on lut_kind

    def _emit_activation(self, p_storage):
        """Compute one tanh value from `p_storage` (storage_ty), return storage_ty."""
        if self.lut_kind == "direct":
            return self._emit_act_direct(p_storage)
        if self.lut_kind == "linear_interp":
            return self._emit_act_linear_interp(p_storage)
        if self.lut_kind == "polynomial":
            return self._emit_act_polynomial(p_storage)
        raise ValueError(f"unknown lut_kind: {self.lut_kind}")

    def _emit_act_direct(self, p_storage):
        g_lut = self.globals["lut_table"]
        idx_i32 = self.b.add(
            self.b.sext(p_storage, _I32), self._ci32(self.lut_offset)
        )
        idx_i64 = self.b.sext(idx_i32, _I64)
        return _load1d_global(self.b, g_lut, idx_i64)

    def _emit_act_linear_interp(self, p_storage):
        """Subsampled table + linear interp, bit-exact mirror of Python ref."""
        g_lut = self.globals["lut_table"]
        f = self.lut_interp_frac_bits
        n = self.lut_n_entries
        # normalized = sext(p, i32) + offset, then t_q = requantize(normalized, idx_M0, idx_n)
        normalized = self.b.add(
            self.b.sext(p_storage, _I32), self._ci32(self.lut_offset)
        )
        t_q = self._emit_requantize_i32(
            normalized, self.lut_idx_M0, self.lut_idx_n
        )
        # idx = t_q >> f, clipped to [0, n-2]
        idx_raw = self.b.ashr(t_q, self._ci32(f))
        zero32 = self._ci32(0)
        n_minus2 = self._ci32(n - 2)
        idx_lo = self.b.select(
            self.b.icmp_signed("<", idx_raw, zero32),
            zero32,
            idx_raw,
        )
        idx = self.b.select(
            self.b.icmp_signed(">", idx_lo, n_minus2),
            n_minus2,
            idx_lo,
        )
        # frac = t_q - (idx << f)
        frac_q = self.b.sub(t_q, self.b.shl(idx, self._ci32(f)))

        # Load y0 = lut[idx], y1 = lut[idx + 1]; widen to i32 for the lerp math.
        idx_i64 = self.b.sext(idx, _I64)
        idx1_i64 = self.b.add(idx_i64, self._ci64(1))
        y0_s = _load1d_global(self.b, g_lut, idx_i64)
        y1_s = _load1d_global(self.b, g_lut, idx1_i64)
        y0_i32 = self.b.sext(y0_s, _I32)
        y1_i32 = self.b.sext(y1_s, _I32)

        # Lerp in i64: y0 + ((y1 - y0) * frac_q) >> f.
        dy_i32 = self.b.sub(y1_i32, y0_i32)
        dy_64 = self.b.sext(dy_i32, _I64)
        frac_64 = self.b.sext(frac_q, _I64)
        scaled_64 = self.b.ashr(self.b.mul(dy_64, frac_64), self._ci64(f))
        interp_i32 = self.b.add(y0_i32, self.b.trunc(scaled_64, _I32))
        return self._emit_saturate_to_storage(interp_i32)

    def _emit_act_polynomial(self, p_storage):
        """Odd-only minimax tanh, Horner in x², bit-exact with Python ref.

        x² = (x·x) >> qf
        inner = ((x²·a5) >> qf) + a3
        outer = ((x²·inner) >> qf) + a1
        y     = (x·outer) >> qf
        y     = clamp(y, ±one_qf)
        """
        qf = self.poly_qf_bits
        # x_qf = requantize(sext(p) - zp_pre, x_M0, x_n), widen to i64.
        centered = self.b.sub(
            self.b.sext(p_storage, _I32), self._ci32(self.zp_pre)
        )
        x_qf_i32 = self._emit_requantize_i32(
            centered, self.poly_x_M0, self.poly_x_n
        )
        x_qf = self.b.sext(x_qf_i32, _I64)
        # Clamp |x| <= x_clip_qf
        clip_pos = self._ci64(self.poly_clip_qf)
        clip_neg = self._ci64(-self.poly_clip_qf)
        x_qf = self.b.select(
            self.b.icmp_signed("<", x_qf, clip_neg), clip_neg, x_qf
        )
        x_qf = self.b.select(
            self.b.icmp_signed(">", x_qf, clip_pos), clip_pos, x_qf
        )
        qf_const = self._ci64(qf)
        a1_const = self._ci64(self.poly_a1_qf)
        a3_const = self._ci64(self.poly_a3_qf)
        a5_const = self._ci64(self.poly_a5_qf)
        # Horner in x²:  y = x · (a1 + x² · (a3 + x² · a5))
        x2_qf = self.b.ashr(self.b.mul(x_qf, x_qf), qf_const)
        inner = self.b.add(
            self.b.ashr(self.b.mul(x2_qf, a5_const), qf_const),
            a3_const,
        )
        outer = self.b.add(
            self.b.ashr(self.b.mul(x2_qf, inner), qf_const),
            a1_const,
        )
        y_qf = self.b.ashr(self.b.mul(x_qf, outer), qf_const)
        # Clamp y to ±one_qf
        one_pos = self._ci64(self.poly_one_qf)
        one_neg = self._ci64(-self.poly_one_qf)
        y_qf = self.b.select(
            self.b.icmp_signed("<", y_qf, one_neg), one_neg, y_qf
        )
        y_qf = self.b.select(
            self.b.icmp_signed(">", y_qf, one_pos), one_pos, y_qf
        )
        # Δq_state = requantize(y_qf), then +zp_state
        y_qf_i32 = self.b.trunc(y_qf, _I32)
        delta = self._emit_requantize_i32(
            y_qf_i32, self.poly_back_M0, self.poly_back_n
        )
        total = self.b.add(delta, self._ci32(self.zp_state))
        return self._emit_saturate_to_storage(total)

    # ------------------------------------------------------------------
    # readout

    def _rq_out(self, x_i32, m, name, M0_scalar, n_scalar):
        """Readout requantize: per-row (M0[m], n[m]) when per_channel_out,
        else the scalar (M0, n). `m` is the output-row SSA index."""
        if self.per_channel_out:
            m0 = _load1d_global(self.b, self.globals[name + "_M0"], m)
            nn = _load1d_global(self.b, self.globals[name + "_n"], m)
            return self._emit_requantize_i32_dynamic(x_i32, m0, nn)
        return self._emit_requantize_i32(x_i32, M0_scalar, n_scalar)

    def _lower_readout_linear(self, op):
        g_Wout = self.globals["W_out"]
        g_rs_state = self.globals["row_sum_Wout_state"]
        g_rs_input = self.globals.get("row_sum_Wout_input")
        F = op.F
        K = self.K
        N = self.N
        Mout = op.M

        off_bias = 0
        off_input = 1 if self.include_bias else 0
        off_state = off_input + (K if self.include_input else 0)

        t = self.t
        tM = self.b.mul(t, _ci(Mout))

        # The readout accumulates in i64 — W_out may be wider than the base
        # storage (mixed precision) and N can be large, so the matmul must
        # not overflow. Each block is clamped to i32 before its requantize,
        # matching the Python reference exactly.
        with _loop(self.b, _ci(Mout), "m") as m:
            y_var = self.b.alloca(_I32, name="y_acc")
            self.b.store(self._ci32(self.zp_output), y_var)

            if self.include_bias:
                w0 = _load2d_global(self.b, g_Wout, F, m, _ci(off_bias))
                clamped_b = self._clamp_i64_to_i32(self.b.sext(w0, _I64))
                rq_b = self._rq_out(
                    clamped_b,
                    m,
                    "M_out_bias",
                    self.M_out_bias_M0,
                    self.M_out_bias_n,
                )
                self.b.store(self.b.add(self.b.load(y_var), rq_b), y_var)

            if self.include_input:
                acc_var = self.b.alloca(_I64, name="acc_input_ro")
                self.b.store(self._ci64(0), acc_var)
                with _loop(self.b, _ci(K), "kin_ro") as k:
                    col = self.b.add(_ci(off_input), k)
                    w = _load2d_global(self.b, g_Wout, F, m, col)
                    x = _load1d(
                        self.b,
                        self.X_arg,
                        self.b.add(self.b.mul(t, _ci(K)), k),
                    )
                    prod = self.b.mul(
                        self.b.sext(w, _I64), self.b.sext(x, _I64)
                    )
                    self.b.store(
                        self.b.add(self.b.load(acc_var), prod),
                        acc_var,
                    )
                rs = self.b.sext(_load1d_global(self.b, g_rs_input, m), _I64)
                adj = self.b.sub(
                    self.b.load(acc_var),
                    self.b.mul(self._ci64(self.zp_input), rs),
                )
                rq_i = self._rq_out(
                    self._clamp_i64_to_i32(adj),
                    m,
                    "M_out_input",
                    self.M_out_input_M0,
                    self.M_out_input_n,
                )
                self.b.store(self.b.add(self.b.load(y_var), rq_i), y_var)

            # State block matvec: vectorize when opt-in + safe. i8 W_out * i8
            # state accumulates in i32 (fast); else i32-multiply -> i64-accum.
            w_out_sty = g_Wout.value_type.element
            use_vec = self.vlen > 1 and self.ro_vec_safe
            ro_acc_ty = (
                _I32
                if (
                    use_vec and w_out_sty.width == 8 and self.storage_bits == 8
                )
                else _I64
            )
            acc_var = self.b.alloca(ro_acc_ty, name="acc_state_ro")
            self.b.store(ir.Constant(ro_acc_ty, 0), acc_var)
            if use_vec:
                w_base = self.b.gep(
                    g_Wout,
                    [
                        _ci32(0),
                        self.b.add(self.b.mul(m, _ci(F)), _ci(off_state)),
                    ],
                )
                self._emit_int_matvec_vec(
                    w_base,
                    self.h_buf,
                    N,
                    acc_var,
                    ro_acc_ty,
                    w_out_sty,
                    self.storage_ty,
                    "jst_ro",
                )
            else:
                with _loop(self.b, _ci(N), "jst_ro") as j:
                    col = self.b.add(_ci(off_state), j)
                    w = _load2d_global(self.b, g_Wout, F, m, col)
                    h = _load1d(self.b, self.h_buf, j)
                    prod = self.b.mul(
                        self.b.sext(w, _I64), self.b.sext(h, _I64)
                    )
                    self.b.store(
                        self.b.add(self.b.load(acc_var), prod), acc_var
                    )
            acc_state_i64 = (
                self.b.load(acc_var)
                if ro_acc_ty == _I64
                else self.b.sext(self.b.load(acc_var), _I64)
            )
            rs = self.b.sext(_load1d_global(self.b, g_rs_state, m), _I64)
            adj = self.b.sub(
                acc_state_i64,
                self.b.mul(self._ci64(self.zp_state), rs),
            )
            rq_s = self._rq_out(
                self._clamp_i64_to_i32(adj),
                m,
                "M_out_state",
                self.M_out_state_M0,
                self.M_out_state_n,
            )
            self.b.store(self.b.add(self.b.load(y_var), rq_s), y_var)

            y_q = self._emit_saturate_to_storage(self.b.load(y_var))
            if self.logits is not None:
                _store1d(self.b, self.logits, m, y_q)
            else:
                _store1d(self.b, self.Y_arg, self.b.add(tM, m), y_q)
