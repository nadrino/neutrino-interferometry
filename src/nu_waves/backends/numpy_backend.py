import numpy as _np
from .interface import ArrayBackend

class _NumpyXP:
    def __init__(self, np):
        self._np = np
    def __getattr__(self, name): return getattr(self._np, name)
    def repeat_last(self, x, n): return self._np.repeat(x[..., self._np.newaxis], n, axis=-1)

class _NumpyLinalg:
    @staticmethod
    def eigh(A):
        return _np.linalg.eigh(A)  # batched
    @staticmethod
    def matrix_exp(A):
        # A: (..., N, N), Hermitian
        w, V = _np.linalg.eigh(A)                 # w: (..., N), V: (..., N, N)
        ew = _np.exp(w)                           # (..., N)
        V_scaled = V * ew[..., _np.newaxis, :]    # scale columns by exp(eigs)
        return V_scaled @ V.conj().swapaxes(-1, -2)

def make_numpy_backend(seed: int | None = None) -> ArrayBackend:
    rng = _np.random.default_rng(seed)
    xp = _NumpyXP(_np)
    return ArrayBackend(
        xp=xp,
        linalg=_NumpyLinalg(),        # <-- use our shim with matrix_exp
        rng=rng,
        dtype_real=_np.float64,
        dtype_complex=_np.complex128,
    )
