from __future__ import annotations
import math
import torch
from .interface import ArrayBackend

class _TorchXP:
    """
    Tiny shim so we can call a few NumPy-like functions from torch.
    Only what VacuumOscillator/Hamiltonian use today.
    """
    def __init__(self, device, dtype_real, dtype_complex):
        self.device = device
        self.dtype_real = dtype_real
        self.dtype_complex = dtype_complex

    def eye(self, N, dtype=None):
        dt = dtype if dtype is not None else self.dtype_complex
        return torch.eye(N, dtype=dt, device=self.device)

    # array-like
    def asarray(self, x, dtype=None):
        if isinstance(x, torch.Tensor):
            t = x
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            return t.to(self.device)
        # numpy / python
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def zeros(self, shape, dtype=None):
        import torch
        dt = dtype if dtype is not None else self.dtype_real
        return torch.zeros(shape, dtype=dt, device=self.device)

    def stack(self, arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    def argmax(self, x, axis=None):
        return torch.argmax(x, dim=axis)

    def angle(self, z):
        # Equivalent of numpy.angle for complex tensors
        return torch.atan2(torch.imag(z), torch.real(z))

    def abs(self, x):         return torch.abs(x)
    def exp(self, x):         return torch.exp(x)
    def conj(self, x):        return torch.conj(x)
    def reshape(self, x, *s): return x.reshape(*s)
    def swapaxes(self, x, a, b): return x.swapaxes(a, b)
    def broadcast_arrays(self, *xs):
        # torch.broadcast_tensors requires tensors, not python scalars
        ts = [x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=self.device) for x in xs]
        return torch.broadcast_tensors(*ts)
    def meshgrid(self, *xs, indexing="ij"):
        return torch.meshgrid(*xs, indexing=indexing)

    def repeat_last(self, x, n):  # used to tile samples along a new last axis
        return x.unsqueeze(-1).repeat(*([1] * x.ndim), n)

    def arange(self, N): return torch.arange(N, device=self.device)

    def einsum(self, subscripts, *operands): return torch.einsum(subscripts, *operands)
    def broadcast_to(self, x, shape): return torch.broadcast_to(x, shape)

    def isscalar(self, x):
        # Behave like numpy.isscalar: True for Python numbers or 0-D tensors
        if isinstance(x, (int, float, complex, bool)):
            return True
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return True
        return False

    def conjugate(self, x):  # alias for NumPy-compat
        return torch.conj(x)

    # numpy-compat helpers used in selection
    def ix_(self, b_idx, a_idx):
        # returns index "grids" for advanced indexing; keep on device
        b = b_idx if isinstance(b_idx, torch.Tensor) else torch.as_tensor(b_idx, dtype=torch.long, device=self.device)
        a = a_idx if isinstance(a_idx, torch.Tensor) else torch.as_tensor(a_idx, dtype=torch.long, device=self.device)
        return torch.meshgrid(b, a, indexing="ij")

    newaxis = None  # not used; `unsqueeze` is used instead

def make_torch_mps_backend(seed: int | None = None, use_complex64: bool = True) -> ArrayBackend:
    if not torch.backends.mps.is_available():
        # fallback to CPU torch; still useful for parity testing
        device = torch.device("cpu")
    else:
        device = torch.device("mps")

    # dtype policy: complex64 on MPS (faster / supported)
    dtype_real    = torch.float32 if use_complex64 else torch.float64
    dtype_complex = torch.complex64 if use_complex64 else torch.complex128

    # RNG on device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    xp = _TorchXP(device=device, dtype_real=dtype_real, dtype_complex=dtype_complex)

    class _TorchLinalg:
        @staticmethod
        def eigh(A):
            try:
                return torch.linalg.eigh(A)  # will raise on MPS
            except NotImplementedError:
                # Fallback: CPU eigh, then move results back to A.device
                dev = A.device
                A_cpu = A.detach().to("cpu")
                w_cpu, V_cpu = torch.linalg.eigh(A_cpu)
                return w_cpu.to(dev), V_cpu.to(dev)

        @staticmethod
        def matrix_exp(A):
            import torch
            try:
                return torch.linalg.matrix_exp(A)  # native if available
            except NotImplementedError:
                dev = A.device
                A_cpu = A.detach().to("cpu")
                S_cpu = torch.linalg.matrix_exp(A_cpu)
                return S_cpu.to(dev)

    class _TorchRNG:
        def __init__(self, generator): self.g = generator
        def normal(self, loc, scale, size):
            # loc/scale can be tensors with broadcastable shapes
            if not isinstance(loc, torch.Tensor):   loc = torch.as_tensor(loc, device=device, dtype=dtype_real)
            if not isinstance(scale, torch.Tensor): scale = torch.as_tensor(scale, device=device, dtype=dtype_real)
            out = torch.empty(size, device=device, dtype=loc.dtype)
            # sample standard normal then affine-transform; faster & avoids missing torch.normal broadcast on MPS versions
            out.normal_(generator=self.g)
            # Broadcast loc/scale to out
            loc = loc.broadcast_to(out.shape)
            scale = scale.broadcast_to(out.shape)
            return out.mul_(scale).add_(loc)

        def uniform(self, low, high, size):
            if not isinstance(low, torch.Tensor):   low = torch.as_tensor(low, device=device, dtype=dtype_real)
            if not isinstance(high, torch.Tensor): high = torch.as_tensor(high, device=device, dtype=dtype_real)
            out = torch.empty(size, device=device, dtype=low.dtype)
            out.uniform_(generator=self.g)
            low  = low.broadcast_to(out.shape)
            high = high.broadcast_to(out.shape)
            return low + (high - low) * out

    rng = _TorchRNG(gen)

    class _TorchBackend(ArrayBackend):
        def to_device(self, x):
            if isinstance(x, torch.Tensor): return x.to(device)
            return torch.as_tensor(x, device=device)
        def from_device(self, x):
            if isinstance(x, torch.Tensor):
                try:   return x.detach().cpu().numpy()
                except Exception: return x.detach().cpu()
            return x

    return _TorchBackend(
        xp=xp,
        linalg=_TorchLinalg,
        rng=rng,
        dtype_real=dtype_real,
        dtype_complex=dtype_complex,
    )
