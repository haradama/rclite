"""RC-aware optimization passes over rclite IR.

A pass is a callable `Module -> Module`. They compose by simple iteration
in `lower_with_passes()`.
"""
from .structural import StructuralSpecialize
from .fuse import FuseStepReadout
from .unroll import TimeUnroll

__all__ = ["StructuralSpecialize", "FuseStepReadout", "TimeUnroll"]
