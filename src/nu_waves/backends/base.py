from dataclasses import dataclass
from typing import Callable


@dataclass
class Backend:
    name: str
    device: str
    dtype_real = np.float64
    dtype_complex = np.complex128
    xp = _NumpyXP()
    linalg = _NumpyLinalg()
    from_device: Callable
