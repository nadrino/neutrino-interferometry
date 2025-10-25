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
        _real_dtype = real_dtype if real_dtype is not None else cls._real_dtype
        _complex_dtype = complex_dtype if complex_dtype is not None else cls._complex_dtype
        if module.__name__ == "torch":
            from nu_waves.backends.torch_backend import TorchBackend
            cls._current_api = TorchBackend()
            if device is not None:
                device = cls._current_api.device(device)
            else:
                print("Auto-selecting torch device")
                if cls._current_api.cuda.is_available():
                    device = cls._current_api.device("cuda")  # NVIDIA GPU
                elif cls._current_api.backends.mps.is_available():
                    device = cls._current_api.device("mps")  # Apple Silicon GPU
                else:
                    device = cls._current_api.device("cpu")  # Fallback

            print("Using device:", device)
            cls._device = device
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


# default is Numpy
Backend.set_api(np)
