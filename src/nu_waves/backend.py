def pick_backend(prefer: str | None = None):
    if prefer == "torch":
        try:
            import torch as xp  # type: ignore
            return xp, "torch"
        except Exception:
            pass
    if prefer == "cupy":
        try:
            import cupy as xp  # type: ignore
            return xp, "cupy"
        except Exception:
            pass
    import numpy as xp
    return xp, "numpy"
