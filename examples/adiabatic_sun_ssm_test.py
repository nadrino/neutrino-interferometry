# examples/solar_mass_fractions_bs05.py
import os
from pathlib import Path
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

from nu_waves.backends import make_torch_mps_backend
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.matter.solar import load_bs05_agsop  # you added this in solar.py
import nu_waves.utils.style

# toggle for CPU/GPU
# torch_backend = None
torch_backend = make_torch_mps_backend(seed=0, use_complex64=True)

URL = "https://www.sns.ias.edu/~jnb/SNdata/Export/BS2005/bs05_agsop.dat"
DATA_DIR = Path("./")
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL = DATA_DIR / "bs05_agsop.dat"

# 1) Download once if needed
if not LOCAL.exists():
    print(f"Downloading BS05 table to {LOCAL} …")
    urllib.request.urlretrieve(URL, LOCAL.as_posix())
    print("Done.")

# 2) Load SSM profile (ρ(r), Ye(r), R_sun)
sol = load_bs05_agsop(LOCAL.as_posix())  # -> SolarProfileSSM

# 3) Oscillator (NumPy backend)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})

osc = Oscillator(
    mixing_matrix=Mixing(dim=3, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_list=spec.get_m2(),
    backend=torch_backend
)

# 4) Physics knobs
E_GeV   = 0.010                      # 10 MeV
r_emit  = 0.05 * sol.R_sun_km        # emission radius
# distances from emission to the surface; include 0 for marker
s_max   = sol.R_sun_km - r_emit
s_km    = np.concatenate(([0.0], np.geomspace(1.0, s_max, 799)))  # (n,)

# 5) Adiabatic evolution from emission
F = osc.adiabatic_mass_fractions_from_emission(
    E_GeV=E_GeV, profile=sol, r_emit_km=r_emit, s_km=s_km, alpha=0, antineutrino=False
)  # shape (n, N)

# Consistency check: initial point equals vacuum_from_matter at emission
F0_vfm = osc.initial_mass_composition(
    alpha=0, basis="vacuum_from_matter", E_GeV=E_GeV, profile=sol, r_emit_km=r_emit
)
# np.testing.assert_allclose(F[0], F0_vfm, atol=1e-10)

# 6) Plot vs radius (log x)
r_path = r_emit + s_km
x = r_path / sol.R_sun_km  # r/R_sun in [~0.05, 1]
labels = [r"$\nu_1$", r"$\nu_2$", r"$\nu_3$"]
colors = ["C0", "C1", "C2"]

plt.figure(figsize=(7.5, 4.2))
for i, (lab, col) in enumerate(zip(labels, colors)):
    plt.plot(x, F[:, i], lw=2, label=lab, color=col)
    # initial and final markers
    plt.scatter([x[0], x[-1]], [F[0, i], F[-1, i]], s=25, color=col, zorder=5)

plt.xscale("log")
plt.xlabel(r"$r/R_\odot$")
plt.ylabel("Mass–state fraction (adiabatic)")
plt.ylim(0, 1.0)
plt.grid(True, which="both", alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("../figures/adiabatic_sun_ssm_test.jpg", dpi=150) if not os.environ.get("CI") else None
plt.show()
