"""Arduino Uno (8-bit AVR / ATmega328P) target for the affine quant path."""

from .target import ArduinoUnoTarget
from .emit_c import emit_affine_kernel_c

__all__ = ["ArduinoUnoTarget", "emit_affine_kernel_c"]
