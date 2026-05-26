"""rclite IR module: the container the lowering visitor walks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .ops import Op


@dataclass
class Module:
    """rclite IR module.

    Holds the top-level op sequence (typically a single `TimeLoop`) and
    the model's weight tensors, referenced by name from ops.
    """
    K: int
    N: int
    M: int
    weights: Dict[str, np.ndarray]
    ops: List[Op]
    metadata: Dict[str, Any] = field(default_factory=dict)
