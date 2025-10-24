# examples/solar_mass_fractions_2MeV.py
import os
from pathlib import Path
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.matter.solar import load_bs05_agsop
from nu_waves.matter.landau import landau_zener_for_pair
import nu_waves.utils.style


# toggle for CPU/GPU
torch_backend = None
# torch_backend = make_torch_mps_backend(seed=0, use_complex64=True)

# ---- data (download once) ----
URL = "https://www.sns.ias.edu/~jnb/SNdata/Export/BS2005/bs05_agsop.dat"
DATA_DIR = Path("data/ssm"); DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL = DATA_DIR / "bs05_agsop.dat"
if not LOCAL.exists():
    print(f"Downloading BS05 to {LOCAL} …")
    urllib.request.urlretrieve(URL, LOCAL.as_posix())

# ---- profile & oscillator ----
sol = load_bs05_agsop(LOCAL.as_posix())          # provides rho_gcm3(r), Ye(r), R_sun_km

# 3) Oscillator (NumPy backend)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

# Masses, normal ordering
spec = Spectrum(n=3, dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428})

osc = Oscillator(
    mixing_matrix=Mixing(dim=3, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_list=spec.get_m2(),
    backend=torch_backend
)

# ---- physics knobs ----
E_GeV  = 0.003      # 2 MeV
r_emit = 0.05 * sol.R_sun_km

# distances from emission to surface; include 0 for initial marker
s_max = sol.R_sun_km - r_emit
s_km  = np.concatenate(([0.0], np.geomspace(1.0, s_max, 799)))
r_path = r_emit + s_km

# ---- adiabatic mass fractions (vacuum basis, phase-averaged) ----
F = osc.adiabatic_mass_fractions_from_emission(
    E_GeV=E_GeV, profile=sol, r_emit_km=r_emit, s_km=s_km, alpha=0, antineutrino=False
)

# initial point (vacuum-from-matter) sanity check
F0 = osc.initial_mass_composition(alpha=0, basis="vacuum_from_matter",
                                  E_GeV=E_GeV, profile=sol, r_emit_km=r_emit)
# This should match F[0] (same eigenvectors at emission)
np.testing.assert_allclose(F[0], F0, rtol=1e-9, atol=1e-12)

# ---- LZ diagnostic for (1,2) crossing: locate avoided crossing and Pc ----
# (You already compute eigenvalues inside; here we recompute along r_path just for the diagnostic.)
xp, linalg = osc.backend.xp, osc.backend.linalg
H_list = []
for rk in r_path:
    rho = float(sol.rho_gcm3([rk])[0])
    Ye  = float(sol.Ye([rk])[0])
    Hk = osc.hamiltonian.matter_constant(E_GeV, rho_gcm3=rho, Ye=Ye, antineutrino=False)
    H_list.append(Hk[0] if Hk.ndim == 3 else Hk)
H = xp.stack(H_list, axis=0)
evals, _ = linalg.eigh(H)                       # (n, N), (n, N, N)
evals_np = osc.backend.from_device(evals)
lz = landau_zener_for_pair(r_path, evals_np, i=0, j=1)  # (1,2) subsystem

# ---- plot ----
x = r_path / sol.R_sun_km
labels = [r"$\nu_1$", r"$\nu_2$", r"$\nu_3$"]; colors = ["C0","C1","C2"]

plt.figure(figsize=(7.6,4.2))
for i,(lab,col) in enumerate(zip(labels, colors)):
    plt.plot(x, F[:, i], lw=2, color=col, label=lab)
    plt.scatter([x[0], x[-1]], [F[0, i], F[-1, i]], s=22, color=col, zorder=5)

# mark the LZ crossing if detected
if lz["has_cross"]:
    xstar = lz["r_star"]/sol.R_sun_km
    plt.axvline(xstar, ls="--", lw=1.2, color="k", alpha=0.3)
    plt.text(xstar*1.02, 0.05, rf"$P_c^{{(12)}}\approx{lz['Pc']:.2g}$", rotation=90,
             va="bottom", ha="left", fontsize=10, alpha=0.8)

plt.xscale("log")
plt.title(f"Neutrino emitted in sun (r/$R_s$={r_emit/sol.R_sun_km}) with energy E={E_GeV*1000:.2g} MeV")
plt.xlabel(r"$r/R_\odot$")
plt.ylabel("Mass–state fraction (adiabatic)")
plt.ylim(0, 1.0)
plt.grid(True, which="both", alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
