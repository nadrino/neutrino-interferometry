import numpy as _np
from .interface import ArrayBackend

class _NumpyXP:
    def __init__(self, np):
        self._np = np
    def __getattr__(self, name): return getattr(self._np, name)
    def repeat_last(self, x, n): return self._np.repeat(x[..., self._np.newaxis], n, axis=-1)

def make_numpy_backend(seed: int | None = None) -> ArrayBackend:
    rng = _np.random.default_rng(seed)
    return ArrayBackend(
        xp=_np,
        linalg=_np.linalg,
        rng=rng,
        dtype_real=_np.float64,
        dtype_complex=_np.complex128,
    )
