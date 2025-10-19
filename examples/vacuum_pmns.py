import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import VacuumOscillator
import nu_waves.utils.flavors as flavors

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.get_mixing_matrix()
print(np.round(U_pmns, 3))

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})
spec.summary()

osc = VacuumOscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2())

# Compute probabilities:
# α = 1 (νμ source), β = [1,2,3] → (νμ, νe, ντ)
E_min, E_max = 0.2, 3.0
Enu_list = np.linspace(E_min, E_max, 200)
P = osc.probability(
    L_km=295, E_GeV=Enu_list,
    alpha=flavors.muon,
    beta=[flavors.electron, flavors.muon, flavors.tau],
    antineutrino=False
)

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.figure(figsize=(6.5, 4.0))

plt.plot(Enu_list, P[:, flavors.electron], label=r"$P_{\mu e}$ appearance", lw=2)
plt.plot(Enu_list, P[:, flavors.muon], label=r"$P_{\mu\mu}$ disappearance", lw=2)
plt.plot(Enu_list, P[:, flavors.tau], label=r"$P_{\mu\tau}$ appearance", lw=2)
plt.plot(Enu_list, P.sum(axis=1), "--", label="Total probability", lw=1.5)

plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel(r"Probability")
plt.title(r"T2K-like vacuum oscillation ($L=295\,\mathrm{km}$)")
plt.xlim(E_min, E_max)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("../figures/vacuum_pmns.pdf")
