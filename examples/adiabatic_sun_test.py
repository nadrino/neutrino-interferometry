import numpy as np
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import VacuumOscillator
from nu_waves.matter.solar import SolarProfile
from nu_waves.backends import make_torch_mps_backend

# toggle for CPU/GPU
# torch_backend = None
torch_backend = make_torch_mps_backend(seed=0, use_complex64=True)

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})

osc = VacuumOscillator(
    mixing_matrix=Mixing(dim=3, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_list=spec.get_m2(),
    backend=torch_backend
)

E_GeV = 0.010  # 10 MeV
sol = SolarProfile()
r0 = 0.01 * sol.R_sun_km
r1 = 1.00 * sol.R_sun_km

# r_grid = np.linspace(r0, r1, 800)
r_grid = np.logspace(np.log10(r0), np.log10(r1), 480)
F = osc.adiabatic_mass_fractions(E_GeV=0.010, profile=sol, r_km=r_grid, alpha=0)

if torch_backend is not None:
    F = torch_backend.from_device(F)

# shape and alignment
assert F.shape == (r_grid.size, 3)


# After calling r_km, F = osc.adiabatic_mass_fractions(...)
labels = [r"$\nu_1$", r"$\nu_2$", r"$\nu_3$"]
colors = ["C0", "C1", "C2"]  # match your lines

# x = (r_grid - r_grid[0]) / 1e5  # distance axis
x = (r_grid) / 1e5  # distance axis

# s = r_grid - r_grid[0]      # km
# x = s / 1e5                 # 1e5 km units

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for i, (c, lab) in enumerate(zip(colors, labels)):
    ax.plot(x, F[:, i], lw=2, color=c, label=lab)
    ax.scatter([x[0], x[-1]], [F[0, i], F[-1, i]], s=25, color=c, zorder=5)

# ax.set_xscale("symlog", linthresh=1e-3)  # linear within |x|<1e-3, log outside
ax.set_xlabel(r"Distance traveled inside Sun [$10^5$ km]")
ax.set_ylabel("Mass–state fraction (adiabatic)")
ax.set_ylim(0, 1.0)
ax.grid(True, which="both", alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout(); plt.show()


