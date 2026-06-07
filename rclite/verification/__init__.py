"""Model verification: structural constraints + input-driven ESP checks."""

from .constraints import (
    ConstraintViolation,
    ESPChecker,
    echo_state_property,
    leak_range,
    density_range,
    WellPosedReservoir,
)

# Input-driven Lyapunov verification requires numpy + runtime; expose only if
# the optional dependency is available so the core IDL stays numpy-free.
try:
    from .input_driven import (
        InputDrivenESPCheck,
        maximum_lyapunov_exponent,
        reservoir_singular_value,
    )
except ImportError:
    pass

__all__ = [
    "ConstraintViolation",
    "ESPChecker",
    "echo_state_property",
    "leak_range",
    "density_range",
    "WellPosedReservoir",
    "InputDrivenESPCheck",
    "maximum_lyapunov_exponent",
    "reservoir_singular_value",
]
