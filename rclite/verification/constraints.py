"""SysML v2: package RC::Constraints"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable

from rclite.core.blocks import ReservoirNode
from rclite.core.profile import Topology


class ConstraintViolation(Exception):
    """Raised when a requirement cannot be satisfied."""


@runtime_checkable
class ESPChecker(Protocol):
    """Duck-typed interface for an empirical ESP check object.

    Implemented by `rc_idl.verification.InputDrivenESPCheck`. Kept as a
    Protocol so that this module stays free of numpy / runtime imports.
    """
    def violations(self) -> List[str]: ...


def echo_state_property(r: ReservoirNode) -> bool:
    """SysML2: constraint def EchoStateProperty (conservative sufficient bound).

    Random reservoirs: rho(W) < 1.
    Structured reservoirs (Rodan-Tino 2011):
      DLR  — nilpotent, always satisfies ESP.
      SCR  — rho(W) = |chain_weight|.
      DLRB — rho(W) bounded by |chain_weight| + |chain_feedback|.
    """
    if r.topology == Topology.DLR:
        return True
    if r.topology == Topology.SCR:
        return abs(r.chain_weight) < 1.0
    if r.topology == Topology.DLRB:
        return abs(r.chain_weight) + abs(r.chain_feedback) < 1.0
    return r.spectral_radius < 1.0


def leak_range(r: ReservoirNode) -> bool:
    """SysML2: constraint def LeakRange { 0 < r.leakRate <= 1 }"""
    return 0.0 < r.leak_rate <= 1.0


def density_range(r: ReservoirNode) -> bool:
    """SysML2: constraint def DensityRange { 0 <= r.density <= 1 }"""
    return 0.0 <= r.density <= 1.0


@dataclass
class WellPosedReservoir:
    """SysML2: requirement def WellPosedReservoir { subject : ReservoirNode }

    By default this enforces the conservative sufficient ESP bound
    `spectral_radius < 1`. When an `empirical_check` is supplied
    (e.g. `rc_idl.verification.InputDrivenESPCheck`), the structural ESP
    bound is replaced by the input-driven Lyapunov-based criterion of
    Yildiz et al. (2012) — strictly stronger evidence in that the actual
    operating trajectory is shown to be contractive. Range checks on
    leak_rate and density always apply.
    """
    subject: ReservoirNode
    empirical_check: Optional[Any] = None  # ESPChecker

    def violations(self) -> List[str]:
        v: List[str] = []
        r = self.subject
        if not leak_range(r):
            v.append(
                f"LeakRange violated: leak_rate={r.leak_rate} not in (0, 1]"
            )
        if not density_range(r):
            v.append(
                f"DensityRange violated: density={r.density} not in [0, 1]"
            )
        if self.empirical_check is None:
            if not echo_state_property(r):
                v.append(
                    f"EchoStateProperty (structural) violated: "
                    f"spectral_radius={r.spectral_radius} >= 1.0 — this is the "
                    f"conservative sufficient bound. Attach an empirical check "
                    f"(rc_idl.verification.InputDrivenESPCheck) to validate "
                    f"the input-driven condition of Yildiz et al. (2012)."
                )
        else:
            v.extend(self.empirical_check.violations())
        return v

    def warnings(self) -> List[str]:
        """Informational notes that do not cause a violation."""
        w: List[str] = []
        r = self.subject
        if self.empirical_check is not None and not echo_state_property(r):
            w.append(
                f"spectral_radius={r.spectral_radius} >= 1.0 violates the "
                f"conservative structural ESP bound — accepted on empirical "
                f"input-driven evidence (Yildiz et al. 2012)."
            )
        return w

    def satisfied(self) -> bool:
        return not self.violations()

    def check(self) -> bool:
        v = self.violations()
        if v:
            raise ConstraintViolation("; ".join(v))
        return True
