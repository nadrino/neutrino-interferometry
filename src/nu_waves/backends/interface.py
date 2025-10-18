# Minimal array-backend interface used by nu_waves numerics.

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ArrayBackend:
    """
    Thin wrapper around a numeric library (NumPy for now; Torch/JAX later).
    """
    xp: any                 # array namespace (np or torch)
    linalg: any             # linalg namespace (np.linalg or torch.linalg)
    rng: any                # random generator (np.random.Generator-like)
    dtype_real: any         # default real dtype
    dtype_complex: any      # default complex dtype

    def to_device(self, x):  # no-op for NumPy; overridden in GPU backends
        return x

    def from_device(self, x):  # no-op for NumPy; overridden in GPU backends
        return x
