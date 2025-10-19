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


@dataclass
class SolarProfileSSM:
    R_sun_km: float
    r_over_R: np.ndarray    # monotonic, in [0, 1]
    rho_gcm3_tab: np.ndarray
    Ye_tab: np.ndarray

    # interpolators that return numpy arrays (backend will cast)
    def rho_gcm3(self, r_km: np.ndarray) -> np.ndarray:
        x = np.asarray(r_km, float) / self.R_sun_km
        x = np.clip(x, self.r_over_R.min(), self.r_over_R.max())
        return np.interp(x, self.r_over_R, self.rho_gcm3_tab)

    def Ye(self, r_km: np.ndarray) -> np.ndarray:
        x = np.asarray(r_km, float) / self.R_sun_km
        x = np.clip(x, self.r_over_R.min(), self.r_over_R.max())
        return np.interp(x, self.r_over_R, self.Ye_tab)


# --- loader for BS05(AGS,OP) -------------------------------------------------
def load_bs05_agsop(path: str, R_sun_km: float = 695_700.0) -> SolarProfileSSM:
    """
    Parse Bahcall & Serenelli BS2005-AGS,OP table (bs05_agsop.dat):
      col2: r/Rsun, col4: rho [g/cm^3],
      col7..12: X(1H), X(4He), X(3He), X(12C), X(14N), X(16O).
    We compute Ye = sum (Z/A)*X_i, and approximate any residual Z/A ≈ 0.5.
    Source: sns.ias.edu ~jnb/SNdata/Export/BS2005/bs05_agsop.dat
    """
    rows = []
    with open(path, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            # keep only rows with ≥12 numeric columns
            if len(parts) >= 12:
                try:
                    vals = [float(p) for p in parts[:12]]
                except ValueError:
                    continue
                rows.append(vals)
    if not rows:
        raise ValueError("Could not parse any numeric rows from BS05 table")

    arr = np.asarray(rows, float)
    r_over_R = arr[:, 1]
    rho = arr[:, 3]

    X_H1   = arr[:, 6]
    X_He4  = arr[:, 7]
    X_He3  = arr[:, 8]
    X_C12  = arr[:, 9]
    X_N14  = arr[:,10]
    X_O16  = arr[:,11]

    # metals not listed explicitly → lump as residual with Z/A ≈ 0.5
    X_sum_listed = X_H1 + X_He4 + X_He3 + X_C12 + X_N14 + X_O16
    X_res = np.clip(1.0 - X_sum_listed, 0.0, 1.0)

    # Ye = Σ (Z/A)*X_i
    Ye = (
        1.0   * X_H1  +           # H-1: Z/A = 1
        0.5   * X_He4 +           # He-4: 2/4
        (2/3) * X_He3 +           # He-3: 2/3
        0.5   * (X_C12 + X_N14 + X_O16) +
        0.5   * X_res             # residual metals approximation
    )

    # ensure monotonic grid for interp
    order = np.argsort(r_over_R)
    return SolarProfileSSM(
        R_sun_km=R_sun_km,
        r_over_R=r_over_R[order],
        rho_gcm3_tab=rho[order],
        Ye_tab=Ye[order],
    )

