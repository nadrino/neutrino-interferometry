import numpy as np

class _NumpyXP:
    def __init__(self):
        # allow x[..., xp.newaxis] in user code
        self.newaxis = np.newaxis

    def __getattr__(self, name):
        # fallback to numpy for anything we didn't wrap explicitly
        if hasattr(np, name):
            return getattr(np, name)
        raise AttributeError(f"_NumpyXP has no attribute {name}")

    def as_idx(self, x, N):
        if x is None:
            return np.arange(N)
        x = np.asarray(x, dtype=int)
        return int(x) if x.ndim == 0 else x

    # common helpers used by the codebase
    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.random.normal(loc=loc, scale=scale, size=size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return np.random.uniform(low=low, high=high, size=size)

    def random(self, size=None):
        return np.random.random(size=size)

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def repeat_last(self, x, n):
        return np.repeat(x[..., np.newaxis], n, axis=-1)

    def isscalar(self, x):
        return np.isscalar(x)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def argmax(self, x, axis=None):
        return np.argmax(x, axis=axis)

    def angle(self, z):
        return np.angle(z)

    def eye(self, n, dtype=None):
        return np.eye(n, dtype=dtype)

    def abs(self, x):
        return np.abs(x)

    def conj(self, x):
        return np.conjugate(x)

    # def conjugate(self, x):
    #     return np.conjugate(x)

    # def exp(self, x):
    #     return np.exp(x)

    def einsum(self, subs, *ops):
        return np.einsum(subs, *ops)


class _NumpyLinalg:
    def eigh(self, A):
        # works for (N,N) or (...,N,N)
        return np.linalg.eigh(A)

    def matrix_exp(self, A):
        """
        Stable numpy implementation without SciPy: eig-based exp.
        Supports (N,N) or (...,N,N).
        """
        A = np.asarray(A)
        if A.ndim == 2:
            w, V = np.linalg.eig(A)
            Vinv = np.linalg.inv(V)
            return (V * np.exp(w)[np.newaxis, :]) @ Vinv

        # batched
        *b, n, _ = A.shape
        A2 = A.reshape((-1, n, n))
        out = np.empty_like(A2, dtype=A.dtype)
        for i in range(A2.shape[0]):
            w, V = np.linalg.eig(A2[i])
            Vinv = np.linalg.inv(V)
            out[i] = (V * np.exp(w)[np.newaxis, :]) @ Vinv
        return out.reshape((*b, n, n))


def make_numpy_backend(seed=0):
    class Backend: ...
    backend = Backend()
    backend.name = "numpy"
    backend.device = "cpu"
    backend.dtype_real = np.float64
    backend.dtype_complex = np.complex128
    backend.xp = _NumpyXP()
    backend.linalg = _NumpyLinalg()
    backend.from_device = lambda x: np.asarray(x)

    def to_device(x): return x
    def from_device(x): return x

    backend.to_device = to_device
    backend.from_device = from_device

    return backend
