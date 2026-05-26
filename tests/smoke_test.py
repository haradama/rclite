"""Stdlib-only smoke tests for the rclite package."""
from __future__ import annotations
import sys
import pathlib
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rclite import (
    InputNode, ReservoirNode, ReadoutNode,
    ReservoirComputer,
    WellPosedReservoir, ConstraintViolation,
    echo_state_property,
    Activation, Distribution, Topology, Trainer,
    Direction, SignalIn, SignalOut, Synapse, WeightMatrix,
    Tensor, TimeSeries, DType,
    Mode, RCMode,
)


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception as e:
        raise AssertionError(
            f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        )
    raise AssertionError(f"Expected {exc_type.__name__}, no exception raised")


def test_valid_esn_construction():
    esn = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="input"),
        reservoir=ReservoirNode(
            units=100, spectral_radius=0.9, leak_rate=0.3,
            density=0.1, seed=42, name="reservoir",
        ),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY, name="readout"),
    )
    assert len(esn.synapses()) == 3
    assert esn.W_in.source is esn.input.out_
    assert esn.W_in.target is esn.reservoir.in_
    assert esn.W_res.source is esn.reservoir.out_
    assert esn.W_res.target is esn.reservoir.in_
    assert esn.W_out.source is esn.reservoir.out_
    assert esn.W_out.target is esn.readout.in_
    assert esn.W_fb is None
    assert esn.W_in.spec.trainable is False
    assert esn.W_out.spec.trainable is True


def test_feedback_synapse_is_created_when_requested():
    esn = ReservoirComputer(
        input=InputNode(units=2, name="in"),
        reservoir=ReservoirNode(units=50, has_feedback=True, name="res"),
        readout=ReadoutNode(units=3, name="out"),
    )
    assert esn.W_fb is not None
    assert esn.reservoir.fb_ is not None
    assert esn.W_fb.source is esn.readout.out_
    assert esn.W_fb.target is esn.reservoir.fb_
    assert len(esn.synapses()) == 4


def test_well_posed_reservoir_passes():
    r = ReservoirNode(units=10, spectral_radius=0.95, leak_rate=0.3, density=0.1)
    WellPosedReservoir(r).check()
    assert WellPosedReservoir(r).satisfied()


def test_well_posed_reservoir_detects_spectral_radius_violation():
    r = ReservoirNode(units=10, spectral_radius=1.5, leak_rate=0.3, density=0.1)
    req = WellPosedReservoir(r)
    assert not req.satisfied()
    violations = req.violations()
    assert any("EchoStateProperty" in v for v in violations)
    expect_raises(ConstraintViolation, req.check)


class _StubEmpirical:
    def __init__(self, violations_list):
        self._v = violations_list
    def violations(self):
        return list(self._v)


def test_well_posed_reservoir_empirical_check_overrides_structural():
    r = ReservoirNode(units=10, spectral_radius=1.5, leak_rate=0.3, density=0.1)
    req = WellPosedReservoir(r, empirical_check=_StubEmpirical([]))
    assert req.satisfied(), req.violations()
    assert any("conservative structural" in w for w in req.warnings())


def test_well_posed_reservoir_empirical_check_can_fail():
    r = ReservoirNode(units=10, spectral_radius=0.9, leak_rate=0.3, density=0.1)
    req = WellPosedReservoir(
        r, empirical_check=_StubEmpirical(["MLE=0.123 >= 0"]),
    )
    assert not req.satisfied()
    expect_raises(ConstraintViolation, req.check)


def test_well_posed_reservoir_range_checks_always_run():
    r = ReservoirNode(units=10, spectral_radius=0.9, leak_rate=0.3, density=0.1)
    r.leak_rate = 1.5  # bypass validator to set illegal value
    req = WellPosedReservoir(r, empirical_check=_StubEmpirical([]))
    assert any("LeakRange" in v for v in req.violations())


def test_invalid_layer_units_rejected():
    expect_raises(ValueError, ReservoirNode, units=0)
    expect_raises(ValueError, ReservoirNode, units=-1)


def test_invalid_leak_rate_rejected():
    expect_raises(ValueError, ReservoirNode, units=10, leak_rate=0.0)
    expect_raises(ValueError, ReservoirNode, units=10, leak_rate=1.5)


def test_invalid_density_rejected():
    expect_raises(ValueError, ReservoirNode, units=10, density=-0.1)
    expect_raises(ValueError, ReservoirNode, units=10, density=1.5)


def test_invalid_washout_rejected():
    expect_raises(ValueError, ReadoutNode, units=1, washout=-1)
    expect_raises(ValueError, ReadoutNode, units=1, regularization=-1.0)


def test_synapse_direction_enforced():
    inp = SignalIn(name="a")
    out = SignalOut(name="b")
    Synapse(source=out, target=inp)
    expect_raises(ValueError, Synapse, source=inp, target=out)
    expect_raises(ValueError, Synapse, source=out, target=out)
    expect_raises(ValueError, Synapse, source=inp, target=inp)


def test_weight_matrix_sparsity_range():
    WeightMatrix(sparsity=0.0)
    WeightMatrix(sparsity=1.0)
    expect_raises(ValueError, WeightMatrix, sparsity=-0.1)
    expect_raises(ValueError, WeightMatrix, sparsity=1.1)


def test_tensor_rank_matches_dim():
    Tensor(rank=2, dim=(3, 4))
    Tensor(rank=2, dim=[3, 4])  # list coerced to tuple
    expect_raises(ValueError, Tensor, rank=1, dim=(3, 4))
    expect_raises(ValueError, Tensor, rank=2, dim=(0, 4))


def test_time_series_dt_positive():
    TimeSeries(rank=1, dim=(10,), dt=0.1)
    expect_raises(ValueError, TimeSeries, rank=1, dim=(10,), dt=0.0)


def test_input_offset_default_is_zero():
    n = InputNode(units=1)
    assert n.input_offset == 0.0
    assert n.input_scaling == 1.0


def test_readout_feature_defaults():
    r = ReadoutNode(units=1)
    assert r.include_bias is True
    assert r.include_input is False


def test_structured_topology_attributes():
    r = ReservoirNode(units=10, topology=Topology.SCR,
                      chain_weight=0.7, chain_feedback=0.0)
    assert r.is_structured()
    assert r.topology == Topology.SCR
    r2 = ReservoirNode(units=10, topology=Topology.RANDOM)
    assert not r2.is_structured()


def test_esp_constraint_topology_aware():
    # DLR is always ESP-satisfying regardless of spectral_radius
    r_dlr = ReservoirNode(units=10, topology=Topology.DLR,
                          spectral_radius=5.0, chain_weight=2.0)
    assert echo_state_property(r_dlr)
    # SCR uses chain_weight, not spectral_radius
    r_scr_ok = ReservoirNode(units=10, topology=Topology.SCR,
                             spectral_radius=99.0, chain_weight=0.9)
    assert echo_state_property(r_scr_ok)
    r_scr_bad = ReservoirNode(units=10, topology=Topology.SCR,
                              spectral_radius=0.1, chain_weight=1.5)
    assert not echo_state_property(r_scr_bad)
    # DLRB uses |chain_weight| + |chain_feedback|
    r_dlrb_ok = ReservoirNode(units=10, topology=Topology.DLRB,
                              chain_weight=0.6, chain_feedback=0.3)
    assert echo_state_property(r_dlrb_ok)
    r_dlrb_bad = ReservoirNode(units=10, topology=Topology.DLRB,
                               chain_weight=0.8, chain_feedback=0.5)
    assert not echo_state_property(r_dlrb_bad)


def test_input_distribution_flows_through_composite():
    rc = ReservoirComputer(
        input=InputNode(units=1, input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=10, topology=Topology.SCR, chain_weight=0.7),
        readout=ReadoutNode(units=1),
    )
    assert rc.W_in.spec.distribution == Distribution.BERNOULLI


def test_readout_online_hyperparameters_validated():
    expect_raises(ValueError, ReadoutNode, units=1, forgetting_factor=0.0)
    expect_raises(ValueError, ReadoutNode, units=1, forgetting_factor=1.5)
    expect_raises(ValueError, ReadoutNode, units=1, learning_rate=0.0)
    expect_raises(ValueError, ReadoutNode, units=1, learning_rate=-0.1)
    expect_raises(ValueError, ReadoutNode, units=1, init_variance=0.0)


def test_rc_mode_state_machine():
    m = RCMode()
    assert m.state == Mode.IDLE
    m.fit()
    assert m.state == Mode.TRAINING
    expect_raises(RuntimeError, m.fit)         # cannot fit while training
    expect_raises(RuntimeError, m.predict)     # cannot predict while training
    m.done()
    assert m.state == Mode.IDLE
    m.predict()
    assert m.state == Mode.INFERRING
    m.done()
    assert m.state == Mode.IDLE
    expect_raises(RuntimeError, m.done)        # cannot signal done from idle


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]


def main() -> int:
    n_pass = n_fail = 0
    for t in TESTS:
        try:
            t()
            print(f"{PASS} {t.__name__}")
            n_pass += 1
        except Exception:
            print(f"{FAIL} {t.__name__}")
            traceback.print_exc()
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed (of {len(TESTS)})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
