from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from typing import Tuple, Callable, Iterable

from nu_waves.utils.units import GF
from nu_waves.sources.sun import R_SUN, SolarModel, SolarSource


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


def matter_potential_from_Ne(Ne_per_cm3: np.ndarray) -> np.ndarray:
    """
    V_e = sqrt(2) * G_F * N_e
    Units: if Ne in 1/cm^3 and G_F in MeV^-2, this returns V in MeV.
    Make sure this matches your existing unit conventions elsewhere.
    """
    # Convert Ne [/cm^3] to [/MeV^3] if you use natural units; otherwise keep coherent with your Hamiltonian.
    # Placeholder: return in "MeV" units consistent with your H.
    return np.sqrt(2.0) * GF * Ne_per_cm3  # adjust if needed


def resonance_Ne_12(Delta_m2_21_eV2: float, theta12_rad: float, E_MeV: np.ndarray) -> np.ndarray:
    """
    N_e(res) = (Delta m^2 cos2θ) / (2 sqrt(2) G_F E)
    Returns electron density at resonance (unit consistent with matter_potential_from_Ne).
    """
    cos2 = np.cos(2*theta12_rad)
    # Convert Δm² [eV²] and E [MeV] coherently to your unit system.
    # If your Hamiltonian uses eV, include appropriate conversion factors.
    # Here we return 'effective Ne' in the same units used in matter_potential_from_Ne.
    return (Delta_m2_21_eV2 * cos2) / (2.0 * np.sqrt(2.0) * GF * E_MeV)


# --- exit mass fractions for (E, r_emit)
def exit_mass_fractions_single(
    osc,
    profile,
    E_GeV: float,
    r_emit_km: float,
    n_steps: int = 512,
    alpha: int = 0,
    antineutrino: bool = False,
) -> np.ndarray:
    """
    Returns p_i at the solar surface (mass-state exit fractions) for one energy and one emission radius.
    Uses your existing adiabatic solver.
    """
    s_end = float(max(R_SUN - r_emit_km, 0.0))
    if s_end == 0.0:
        # Emission at surface: already in vacuum eigenbasis
        F_last = np.zeros(osc._n_neutrinos)
        F_last[:] = 0.0
        F_last[0] = 1.0  # if you emitted a pure νe at the surface
        return F_last

    s_km = np.linspace(0.0, s_end, int(n_steps))
    F = osc.adiabatic_mass_fractions_from_emission(
        E_GeV=E_GeV,
        profile=profile,
        r_emit_km=r_emit_km,
        s_km=s_km,
        alpha=alpha,
        antineutrino=antineutrino,
    )  # shape (n_steps, Nstates)
    return np.asarray(F[-1], dtype=float)  # (Nstates,)


# --- Production-radius nodes/weights from SolarModel radial pdf
def radial_nodes_and_weights_from_pdf(
    model: SolarModel,
    source: SolarSource,
    n_nodes: int = 48,
    r_grid_size: int = 6000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds importance-sampled nodes via inverse-CDF over r in [0, R_SUN].
    Returns:
      r_nodes_km: (n_nodes,) radii
      w_nodes:    (n_nodes,) probabilities, sum to 1
    """
    r = np.linspace(0.0, R_SUN, r_grid_size)
    fr = model.radial_pdf(source, r)
    # CDF via trapezoid integration (stable in core)
    cdf = np.cumsum(0.5 * (fr[1:] + fr[:-1]) * np.diff(r))
    cdf = np.concatenate([[0.0], cdf])
    if cdf[-1] == 0.0:
        raise RuntimeError("Radial pdf normalization is zero.")
    cdf /= cdf[-1]
    # Equal-probability bins → midpoints as nodes
    q_edges = np.linspace(0.0, 1.0, n_nodes + 1)
    r_edges = np.interp(q_edges, cdf, r)
    r_nodes = 0.5 * (r_edges[:-1] + r_edges[1:])
    w_nodes = np.diff(q_edges)  # sums to 1
    return r_nodes, w_nodes

# --- Average p_i over production radius for a vector of energies
def exit_mass_fractions_averaged_over_radius(
    osc,
    profile,
    E_GeV: np.ndarray,
    r_nodes_km: np.ndarray,
    w_nodes: np.ndarray,
    n_steps_per_ray: int = 50,
) -> np.ndarray:
    """
    Returns p_mass(E,i) of shape (len(E_GeV), Nstates),
    averaged over the provided (r_nodes, w_nodes).
    """
    E_GeV = np.asarray(E_GeV, dtype=float)
    nE = E_GeV.size
    # infer number of states from osc
    # Assuming osc.dim exists; else use len(osc.m2_list)
    nstates = getattr(osc, "dim", len(osc.m2_list))
    out = np.zeros((nE, nstates), dtype=float)
    for ie, E in tqdm(enumerate(E_GeV), total=nE, desc=f"Processing {E} MeV"):
        acc = np.zeros(nstates, dtype=float)
        for r, w in zip(r_nodes_km, w_nodes):
            p = exit_mass_fractions_single(
                osc, profile, E, r, n_steps=n_steps_per_ray, alpha=0, antineutrino=False
            )
            acc += w * p
        out[ie, :] = acc
    return out  # (nE, nstates)

# --- Daytime Pee from exit fractions and U
def pee_day_from_exit_fractions(
    p_mass: np.ndarray,  # (nE, nstates)
    U: np.ndarray,       # mixing matrix (flavor x mass)
) -> np.ndarray:
    """
    Daytime / vacuum-averaged propagation from Sun to Earth:
      Pee(E) = sum_i p_i(E) * |U_ei|^2
    """
    Uei2 = np.abs(U[0, :]) ** 2
    return (p_mass * Uei2[None, :]).sum(axis=1)

# --- Sanity-limit hlpers for plotting bands
def pee_lowE_vacuum_avg(U: np.ndarray) -> float:
    """ Low-E (vacuum-averaged) limit: Σ_i |U_ei|^4 """
    row = np.abs(U[0, :]) ** 2
    return float((row ** 2).sum())

def pee_highE_msw(U: np.ndarray, mass_index: int = 1) -> float:
    """
    High-E adiabatic LMA limit ~ |U_e2|^2 in NO.
    'mass_index=1' is the column for ν2 if your ordering is [1,2,3].
    """
    return float(np.abs(U[0, mass_index]) ** 2)
