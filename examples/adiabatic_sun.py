import numpy as np
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.matter.solar import SolarProfile
from nu_waves.backends import make_torch_backend

# toggle for CPU/GPU
torch_backend = None
# torch_backend = make_torch_mps_backend(seed=0, use_complex64=True)

# 3 flavors PMNS, PDG values (2025)
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


sol = SolarProfile()

E_GeV = 0.010  # 10 MeV

# neutrino start
r_emit = 0.01 * sol.R_sun_km
s = np.concatenate(([0.0], np.geomspace(1.0, sol.R_sun_km - r_emit, 799)))  # km (include 0)

F0_vac = osc.initial_mass_composition(alpha=0, basis="vacuum")
print(f"F0 vacuum: {F0_vac}")
F0_m   = osc.initial_mass_composition(alpha=0, basis="matter", E_GeV=0.010, profile=sol, r_emit_km=r_emit)
print(f"F0 m: {F0_m}")
F0_vfm = osc.initial_mass_composition(alpha=0, basis="vacuum_from_matter", E_GeV=0.010, profile=sol, r_emit_km=r_emit)
print(f"F0 vfm: {F0_vfm}")

F = osc.adiabatic_mass_fractions_from_emission(
    E_GeV=E_GeV,
    profile=sol,
    r_emit_km=r_emit,
    s_km=s,
    alpha=0
)

# plot (log radius or symlog distance—your pick)
x = s / 1e5
import matplotlib.pyplot as plt
cols = ["C0","C1","C2"]; labs = [r"$\nu_1$", r"$\nu_2$", r"$\nu_3$"]
plt.figure(figsize=(7.5,4.2))
for i,(c,lab) in enumerate(zip(cols,labs)):
    plt.plot(x, F[:, i], lw=2, color=c, label=lab)
    # plt.scatter([x[0]], [F[0, i]], color=c, s=25)  # initial (vacuum-basis at r_emit)
    # plt.scatter([x[0]], [F0_vac[i]], color=c, s=25)  # initial (vacuum-basis at r_emit)
    # plt.scatter([x[0]], [F0_m[i]], color=c, s=25)  # initial (vacuum-basis at r_emit)
    plt.scatter([x[0]], [F0_vfm[i]], color=c, s=25)  # initial (vacuum-basis at r_emit)
plt.xlabel(r"Distance from emission [$10^5$ km]"); plt.ylabel("Mass–state fraction")
plt.ylim(0,1); plt.legend(frameon=False); plt.grid(True, which="both", alpha=0.3)
plt.show()

print("Initial (matter-basis) weights:", F0_vfm)
print("Initial (vacuum-basis)  weights:", F0_vac)

