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
        return _np.linalg.eigh(A)  # batched OK

    @staticmethod
    def matrix_exp(A):
        """
        Exp for skew-Hermitian A (our use case: A = -i * H * L, with H Hermitian).
        Uses: exp(A) = V diag(exp(-i * w)) V^† where iA = V diag(w) V^†, w real.
        """
        B = 1j * A                                # iA is Hermitian
        w, V = _np.linalg.eigh(B)                 # (..., N), (..., N, N)
        ew = _np.exp(-1j * w)                     # (..., N), complex
        V_scaled = V * ew[..., _np.newaxis, :]    # scale columns
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
