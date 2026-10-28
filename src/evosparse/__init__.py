"""EvoSparse: dynamic sparse attention across decoding steps and layers."""

from .attention import (
    enable_evosparse,
    enable_evosparse_without_cross_layer,
    enable_evosparse_without_cross_step,
    reset_evosparse_state,
)

__all__ = [
    "enable_evosparse",
    "enable_evosparse_without_cross_layer",
    "enable_evosparse_without_cross_step",
    "reset_evosparse_state",
]
