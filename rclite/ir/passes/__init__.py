"""RC-aware optimization passes over rclite IR.

A pass is a callable `Module -> Module`. They compose by simple iteration
in `lower_with_passes()`.
"""

from .structural import StructuralSpecialize
from .fuse import FuseStepReadout
from .unroll import TimeUnroll
from .sparsify import SparsifyReservoir, sparse_passes
from .stability import NormalizeReservoir, VerifyEchoStateConstraint
from .prune import PruneInactiveNodes
from .profile import ProfileReservoir
from .refit import RefitReadout

__all__ = [
    "StructuralSpecialize",
    "FuseStepReadout",
    "TimeUnroll",
    "SparsifyReservoir",
    "sparse_passes",
    "NormalizeReservoir",
    "VerifyEchoStateConstraint",
    "PruneInactiveNodes",
    "ProfileReservoir",
    "RefitReadout",
]
