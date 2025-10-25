import torch


class TorchBackend:

    """Subset of Array-API interface mapped to torch."""
    def __getattr__(self, name):
        # delegate to torch if it exists
        if hasattr(torch, name):
            return getattr(torch, name)
        raise AttributeError(f"TorchCompat: torch has no attribute '{name}'")

    # explicit overrides for missing Array API functions
    def asarray(self, x, dtype=None):
        return torch.as_tensor(x, dtype=dtype)

    def copy(self, x):
        return x.clone()

    def conjugate(self, x):
        return torch.conj(x)

    def ndim(self, x):
        """Return number of dimensions of tensor."""
        return x.ndim

    def shape(self, x):
        """Return shape tuple."""
        return tuple(x.shape)

    def size(self, x, dim=None):
        """Return total number of elements or along a given dimension."""
        if dim is not None:
            return x.shape[dim]
        # flatten torch.Size to int
        return int(torch.tensor(x.numel()).item())

    def repeat(self, a, repeats, axis=None):
        """NumPy-style repeat for torch backend."""
        import torch
        if not torch.is_tensor(a):
            a = torch.as_tensor(a)

        if axis is None:
            # Flatten then repeat globally
            a_flat = a.flatten()
            return a_flat.repeat_interleave(repeats)

        # Axis-specific repeat
        if isinstance(repeats, int):
            return a.repeat_interleave(repeats, dim=axis)
        else:
            # repeats is sequence of per-element counts
            if len(repeats) != a.size(axis):
                raise ValueError("len(repeats) must match dimension length")
            # Build index for repeat_interleave
            idx = torch.arange(a.size(axis), device=a.device).repeat_interleave(
                torch.as_tensor(repeats, device=a.device))
            return torch.index_select(a, axis, idx)

    class _Random:
        def standard_normal(self, shape):
            from nu_waves.globals.backend import Backend
            device = Backend.device() if hasattr(Backend, "device") else "cpu"
            return torch.randn(shape, device=device)

    random = _Random()

