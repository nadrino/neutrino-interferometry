# src/nu_waves/matter/supernova.py
import numpy as np
from nu_waves.utils.units import KM_TO_EVINV
from nu_waves.utils.units import VCOEFF_EV

# Practical conversion: V_e [eV] ≈ 7.63e-14 * (Y_e * rho[g/cm^3])


class CoreCollapseSN:
    """
    ρ(r) = ρ0 * (r0 / r)^n * S(r),  r in km, ρ in g/cm^3.
    S(r) is a smooth shock step: 1 (inside) → f (outside) across width Δr.
    """
    def __init__(
        self, rho0=1e12, r0_km=10.0, n=3.0,
        shock_radius_km=7000.0, shock_width_km=10.0, shock_drop=0.2,  # narrow → non-adiabatic
        Ye=0.5,
    ):
        self.rho0 = float(rho0)
        self.r0 = float(r0_km)
        self.n = float(n)
        self.rs = float(shock_radius_km)
        self.dw = max(float(shock_width_km), 1e-6)
        self.f = float(shock_drop)  # (0<f<=1) density factor after shock
        self.Ye = float(Ye)

    # ---------- profile ----------
    def _shock_factor(self, r_km):
        x = (self.rs - np.asarray(r_km)) / self.dw
        # 1 (r<<rs) → f (r>>rs)
        return 0.5 * (1 + self.f) + 0.5 * (1 - self.f) * np.tanh(x)

    def rho(self, r_km):
        r = np.asarray(r_km, dtype=float)
        base = self.rho0 * (self.r0 / r) ** self.n
        return base * self._shock_factor(r)

    def Ve(self, r_km):
        return VCOEFF_EV * self.Ye * self.rho(r_km)

    def dlnNe_dr(self, r_km, h_km=0.5):
        """d ln N_e / dr at r (1/km). Ye const → same as d ln ρ / dr."""
        r = float(r_km)
        r1, r2 = max(r - h_km, 1.1), r + h_km
        y1, y2 = self.rho(r1), self.rho(r2)
        return (np.log(y2) - np.log(y1)) / (r2 - r1)

    # ---------- resonances ----------
    def _find_root_on_grid(self, target_Ve, rmin=20.0, rmax=1.5e5, n=6000):
        r = np.geomspace(rmin, rmax, n)
        y = self.Ve(r) - target_Ve
        s = np.sign(y)
        ix = np.where(s[:-1] * s[1:] < 0)[0]
        if ix.size == 0:
            return None
        k = ix[0]
        r1, r2 = r[k], r[k + 1]
        y1, y2 = y[k], y[k + 1]
        return float(r1 + (r2 - r1) * (-y1) / (y2 - y1))

    def resonance_radius_H(self, dm31_eV2, theta13_rad, E_MeV):
        E = 1e6 * E_MeV
        target = (dm31_eV2 * np.cos(2 * theta13_rad)) / (2.0 * E)   # Δ/2E · cos2θ13
        return self._find_root_on_grid(target)

    def resonance_radius_L(self, dm21_eV2, theta12_rad, theta13_rad, E_MeV):
        E = 1e6 * E_MeV
        target = (dm21_eV2 * np.cos(2 * theta12_rad)) / (2.0 * E) / (np.cos(theta13_rad) ** 2)
        return self._find_root_on_grid(target)

    # ---------- Landau–Zener via Parke γ ----------
    def parke_Pc(self, which, E_MeV, dm21, dm31, th12, th13):
        """
        Pc = exp(-π γ / 2), γ = (Δm^2/(2E)) * (sin^2 2θ / cos 2θ) * (1/|d ln N_e/dr|_res).
        Returns (Pc, r_res). Units consistent via KM_TO_eVINV factor.
        """
        E = 1e6 * E_MeV
        if which.upper() == "H":
            r_res = self.resonance_radius_H(dm31, th13, E_MeV)
            if r_res is None: return 0.0, None
            gamma = (dm31 / (2.0 * E)) * (np.sin(2 * th13) ** 2 / np.cos(2 * th13))
        elif which.upper() == "L":
            r_res = self.resonance_radius_L(dm21, th12, th13, E_MeV)
            if r_res is None: return 0.0, None
            gamma = (dm21 / (2.0 * E)) * (np.sin(2 * th12) ** 2 / np.cos(2 * th12)) * (np.cos(th13) ** 2)
        else:
            raise ValueError("which must be 'H' or 'L'")
        dlnN = abs(self.dlnNe_dr(r_res))
        gamma *= KM_TO_EVINV / max(dlnN, 1e-30)  # convert 1/km → 1/eV
        return float(np.exp(-0.5 * np.pi * gamma)), r_res
