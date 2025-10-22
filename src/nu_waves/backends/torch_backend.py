# torch_backend.py
import torch
import numpy as np

def _map_dtype_torch(requested, dtype_real, dtype_complex):
    if requested is None:
        return dtype_real
    # ---- integers (FIX) ----
    if requested in (int, np.int64, torch.int64):
        return torch.int64
    if requested in (np.int32, torch.int32):
        return torch.int32
    # ---- floats ----
    if requested in (float, np.float32, torch.float32):
        return torch.float32
    if requested in (np.float64, torch.float64):
        return torch.float64
    # ---- complex ----
    if requested in (complex, np.complex64, torch.complex64):
        return torch.complex64
    if requested in (np.complex128, torch.complex128):
        return torch.complex128
    # fallback to backend defaults
    return dtype_real


class _TorchXP:
    def __init__(self, device, dtype_real, dtype_complex):
        self.device = device
        self.dtype_real = dtype_real
        self.dtype_complex = dtype_complex
        # allow x[..., xp.newaxis] indexing
        self.newaxis = None

    def __getattr__(self, name):
        # delegate missing ops to torch when possible
        if hasattr(torch, name):
            return getattr(torch, name)
        raise AttributeError(f"_TorchXP has no attribute {name}")


    def shape(self, x, N):
        if x is None:
            return torch.arange(N)
        x = torch.asarray(x, dtype=int)
        return int(x) if x.ndim == 0 else x

    def as_idx(self, x, N):
        if x is None:
            return torch.arange(N)
        x = torch.asarray(x, dtype=int)
        return int(x) if x.ndim == 0 else x

    def ndim(self, x):
        """Return number of dimensions of a Torch tensor, NumPy array, or scalar."""
        # handle scalar / list / numpy input gracefully
        if isinstance(x, torch.Tensor):
            return x.ndim
        elif hasattr(x, "ndim"):
            return x.ndim
        else:
            # fallback for numbers or lists
            try:
                return torch.as_tensor(x).ndim
            except Exception:
                return 0

    def astype(self, x, dtype, copy=False):
        return x.to(dtype=dtype)

    def size(self, x):
        """Return total number of elements in a Torch tensor or array-like."""
        if isinstance(x, torch.Tensor):
            return x.numel()
        elif hasattr(x, "size"):
            # NumPy or other array-likes
            val = x.size
            return val() if callable(val) else val
        else:
            try:
                return torch.as_tensor(x).numel()
            except Exception:
                return 1

    def maximum(self, x, y):
        # torch.maximum only works on tensors — make sure both are tensors
        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=self.device)
        y_t = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, device=self.device)
        return torch.maximum(x_t, y_t)

    def minimum(self, x, y):
        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=self.device)
        y_t = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, device=self.device)
        return torch.minimum(x_t, y_t)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """
        Torch equivalent of np.random.normal, supporting tensor broadcasting
        and explicit size expansion (unlike torch.normal).
        """
        # Convert numpy arrays to tensors if needed
        if not isinstance(loc, torch.Tensor):
            loc = torch.as_tensor(loc, device=self.device)
        if not isinstance(scale, torch.Tensor):
            scale = torch.as_tensor(scale, device=self.device)

        # Case 1: both tensors
        if isinstance(loc, torch.Tensor) and isinstance(scale, torch.Tensor):
            if size is None or tuple(size) == tuple(loc.shape):
                # Default: same shape as loc
                return torch.normal(mean=loc, std=scale)
            else:
                # Manually expand via broadcasting to requested size
                loc_b = loc.expand(size)
                scale_b = scale.expand(size)
                return torch.normal(mean=loc_b, std=scale_b)

        # Case 2: at least one scalar → use size directly
        mean_val = float(loc) if not isinstance(loc, torch.Tensor) else float(loc)
        std_val  = float(scale) if not isinstance(scale, torch.Tensor) else float(scale)
        return torch.normal(mean=mean_val, std=std_val, size=size, device=self.device)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return (high - low) * torch.rand((), device=self.device) + low
        return (high - low) * torch.rand(size, device=self.device) + low

    def random(self, size=None):
        return torch.rand(size, device=self.device)

    def broadcast_arrays(self, *xs):
        # torch.broadcast_tensors requires tensors, not python scalars
        ts = [x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=self.device) for x in xs]
        return torch.broadcast_tensors(*ts)

    def asarray(self, x, dtype=None):
        dt = _map_dtype_torch(dtype, self.dtype_real, self.dtype_complex)
        return torch.as_tensor(x, dtype=dt, device=self.device)

    def repeat_last(self, x, n):
        # x[..., None].repeat(..., n) would also work
        return x.unsqueeze(-1).repeat_interleave(n, dim=-1)

    def isscalar(self, x):
        return not isinstance(x, torch.Tensor) or x.ndim == 0

    def stack(self, arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    def argmax(self, x, axis=None):
        return torch.argmax(x, dim=axis)

    def angle(self, z):
        return torch.atan2(torch.imag(z), torch.real(z))

    def eye(self, n, dtype=None):
        dt = _map_dtype_torch(dtype, self.dtype_real, self.dtype_complex)
        return torch.eye(n, dtype=dt, device=self.device)

    def abs(self, x):
        return torch.abs(x)

    def conj(self, x):
        return torch.conj(x)

    def conjugate(self, x):
        return torch.conj(x)

    def exp(self, x):
        return torch.exp(x)

    def einsum(self, subs, *ops):
        return torch.einsum(subs, *ops)

    def zeros(self, shape, dtype=None):
        dt = _map_dtype_torch(dtype, self.dtype_real, self.dtype_complex)
        return torch.zeros(shape, dtype=dt, device=self.device)

    def ones(self, shape, dtype=None):
        dt = _map_dtype_torch(dtype, self.dtype_real, self.dtype_complex)
        return torch.ones(shape, dtype=dt, device=self.device)

    def full(self, shape, fill_value, dtype=None):
        dt = _map_dtype_torch(dtype, self.dtype_real, self.dtype_complex)
        return torch.full(shape, fill_value, dtype=dt, device=self.device)

    def zeros_like(self, x, dtype=None):
        dt = _map_dtype_torch(dtype, x.dtype, self.dtype_complex if x.is_complex() else self.dtype_real)
        return torch.zeros_like(x, dtype=dt, device=x.device)

    def ones_like(self, x, dtype=None):
        dt = _map_dtype_torch(dtype, x.dtype, self.dtype_complex if x.is_complex() else self.dtype_real)
        return torch.ones_like(x, dtype=dt, device=x.device)

    def diag(self, v, diagonal=0):
        # ensure v is on device first
        vt = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=self.device)
        return torch.diag(vt, diagonal=diagonal)


class _TorchLinalg:
    def __init__(self, device, dtype_real, dtype_complex):
        self.device = device
        self.dtype_real = dtype_real
        self.dtype_complex = dtype_complex

    def eigh(self, A):
        """
        Deterministic & stable Hermitian eigendecomposition:
        run on CPU in float64/complex128, then cast back.
        Works for (N,N) or (...,N,N).
        """
        A_cpu = A.to("cpu")
        if A_cpu.dtype.is_complex:
            A_cpu = A_cpu.to(torch.complex128)
        else:
            A_cpu = A_cpu.to(torch.float64)

        w_cpu, V_cpu = torch.linalg.eigh(A_cpu)  # batched OK
        w = w_cpu.to(self.device, dtype=self.dtype_real)
        V = V_cpu.to(self.device, dtype=self.dtype_complex)
        return w, V

    def matrix_exp(self, A):
        """
        Use CPU fallback in float64/complex128 for numerical stability
        and MPS compatibility; cast result back to original device/dtype.
        """
        A_cpu = A.to("cpu")
        if A_cpu.dtype.is_complex:
            A_cpu = A_cpu.to(torch.complex128)
        else:
            A_cpu = A_cpu.to(torch.float64)
        Y_cpu = torch.linalg.matrix_exp(A_cpu)
        return Y_cpu.to(self.device, dtype=A.dtype)


def make_torch_backend(seed=0, use_complex64=True, force_device=None):
    torch.manual_seed(seed)

    if force_device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(force_device)

    dtype_real    = torch.float32 if use_complex64 else torch.float64
    dtype_complex = torch.complex64 if use_complex64 else torch.complex128

    class Backend: ...
    backend = Backend()
    backend.name = "torch"
    backend.device = device
    backend.dtype_real = dtype_real
    backend.dtype_complex = dtype_complex
    backend.xp = _TorchXP(device, dtype_real, dtype_complex)
    backend.linalg = _TorchLinalg(device, dtype_real, dtype_complex)

    def to_device(x):
        return torch.as_tensor(x, device=device)

    def from_device(x):
        # detach is safe for tensors; passthrough for numpy scalars
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    backend.to_device = to_device
    backend.from_device = from_device
    return backend
