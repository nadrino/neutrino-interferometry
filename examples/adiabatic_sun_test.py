import numpy as np
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import VacuumOscillator
from nu_waves.matter.solar import SolarProfile

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})

osc = VacuumOscillator(
    mixing_matrix=Mixing(dim=3, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_list=spec.get_m2()
)

E_GeV = 0.010  # 10 MeV
sol = SolarProfile()
r0 = 0.05 * sol.R_sun_km
r1 = 1.00 * sol.R_sun_km

r_grid = np.linspace(0.05*sol.R_sun_km, 1.0*sol.R_sun_km, 800)
F = osc.adiabatic_mass_fractions(E_GeV=0.010, profile=sol, r_km=r_grid, alpha=0)

# shape & alignment
assert F.shape == (r_grid.size, 3)


# After calling r_km, F = osc.adiabatic_mass_fractions(...)
s_km = r_grid - r_grid[0]
labels = [r"$\nu_1$", r"$\nu_2$", r"$\nu_3$"]
colors = ["C0", "C1", "C2"]  # match your lines

x = (r_grid - r_grid[0]) / 1e5  # distance axis

plt.figure(figsize=(7.5,4.2))
for i, (lab, col) in enumerate(zip([r"$\nu_1$", r"$\nu_2$", r"$\nu_3$"], ["C0","C1","C2"])):
    plt.plot(x, F[:, i], lw=2, color=col, label=lab)
    # initial and final markers
    plt.scatter([x[0], x[-1]], [F[0, i], F[-1, i]], color=col, s=25, zorder=5)

plt.xlabel(r"Distance traveled inside Sun [$10^5$ km]")
plt.ylabel("Mass–state fraction (adiabatic)")
plt.ylim(0, 1.0)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

