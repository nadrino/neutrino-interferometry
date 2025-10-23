from array_api_compat import get_namespace
import numpy as np


# static class
class Backend:
    _current_api = np  # default
    _real_dtype = "float64"
    _complex_dtype = "complex128"
    _device = None  # e.g. "cuda", "cpu", "mps"

    @classmethod
    def set_api(cls, module, device=None, real_dtype=None, complex_dtype=None):
        """Set xp backend: numpy, torch, cupy, jax.numpy, etc."""
        import numpy as np
        import torch

        if module is torch:
            cls._current_api = TorchCompat()
        else:
            cls._current_api = module

    @classmethod
    def xp(cls):
        """Return the current array API namespace."""
        return cls._current_api

    @classmethod
    def real_dtype(cls):
        xp = cls._current_api
        return getattr(xp, cls._real_dtype, cls._real_dtype)

    @classmethod
    def complex_dtype(cls):
        xp = cls._current_api
        return getattr(xp, cls._complex_dtype, cls._complex_dtype)

    @classmethod
    def to_device(cls, arr):
        """Move an array to the configured device if supported."""
        xp = cls._current_api
        if hasattr(xp, "asarray") and cls._device:
            if "torch" in xp.__name__:
                return arr.to(cls._device)
            elif "cupy" in xp.__name__:
                return xp.asarray(arr)  # CuPy arrays already live on GPU
            # JAX: handled automatically
        return arr

    @classmethod
    def from_device(cls, arr):
        """Pull array back to CPU (NumPy)."""
        xp = cls._current_api
        if "torch" in xp.__name__:
            return arr.detach().cpu().numpy()
        elif "cupy" in xp.__name__:
            import cupy as cp
            return cp.asnumpy(arr)
        elif "jax" in xp.__name__:
            import jax.numpy as jnp
            import numpy as np
            return np.asarray(jnp.array(arr))
        return arr

class TorchCompat:

    """Subset of Array-API interface mapped to torch."""
    def __getattr__(self, name):
        import torch
        # delegate to torch if it exists
        if hasattr(torch, name):
            return getattr(torch, name)
        raise AttributeError(f"TorchCompat: torch has no attribute '{name}'")

    # explicit overrides for missing Array API functions
    def asarray(self, x, dtype=None):
        import torch
        return torch.as_tensor(x, dtype=dtype)

    def conjugate(self, x):
        import torch
        return torch.conj(x)

    def ndim(self, x):
        """Return number of dimensions of tensor."""
        return x.ndim

    def shape(self, x):
        """Return shape tuple."""
        return tuple(x.shape)

    def size(self, x, dim=None):
        """Return total number of elements or along a given dimension."""
        import torch
        if dim is not None:
            return x.shape[dim]
        # flatten torch.Size to int
        return int(torch.tensor(x.numel()).item())


# default is Numpy
Backend.set_api(np)
