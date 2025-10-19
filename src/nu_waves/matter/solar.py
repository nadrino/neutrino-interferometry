from dataclasses import dataclass
import numpy as np


@dataclass
class SolarProfile:
    R_sun_km: float = 695_700.0

    # User-supplied callables; defaults are simple exponentials (placeholder)
    # You can swap these with an SSM table/fit later.
    def rho_gcm3(self, r_km: np.ndarray) -> np.ndarray:
        x = np.asarray(r_km) / self.R_sun_km
        return 150.0 * np.exp(-10.54 * x)  # crude core→surface falloff

    def Ye(self, r_km: np.ndarray) -> np.ndarray:
        # nearly hydrogen/helium mix; small variation with r
        return np.full_like(np.asarray(r_km, float), 0.5)

    def grid(self, r0_km: float, r1_km: float, n: int) -> np.ndarray:
        return np.linspace(r0_km, r1_km, n)
