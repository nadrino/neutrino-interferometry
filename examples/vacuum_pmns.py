import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.propagation.oscillator import VacuumOscillator
import nu_waves.propagation.solvers as solvers


# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49.0)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.get_mixing_matrix()
print(np.round(U_pmns, 3))


# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({
    (2, 1): 7.42e-5,
    (3, 2): 0.0024428
})
spec.summary()
m2_diag = np.diag(spec.get_m2())


hamiltonian = Hamiltonian(mixing_matrix=U_pmns, m2_diag=m2_diag)
P_dis_test = solvers.probability_alpha_to_beta(
    solvers.propagator_vacuum(
        hamiltonian.vacuum(0.6),
        295
    ),
    alpha=2, beta=2
)
print("P(survival nu_mu, 0.6 GeV, 295 km)", P_dis_test)

E_min, E_max = 0.2, 3.0
Enu_list = np.linspace(E_min, E_max, 200)
P_dis = solvers.probability_alpha_to_beta(
    solvers.propagator_vacuum(
        hamiltonian.vacuum(Enu_list),
        295
    ),
    alpha=2, beta=2
)
plt.figure(figsize=(6,4))
plt.plot(Enu_list, P_dis, lw=2)
plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel(r"$P(\nu_\mu \to \nu_\mu)$")
plt.title("Vacuum oscillation probability")
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Compute probabilities:
# α = 1 (νμ source), β = [1,2,3] → (νμ, νe, ντ)
osc = VacuumOscillator(mixing_matrix=U_pmns, m2_diag=m2_diag)
P = osc.probability(L_km=295, E_GeV=Enu_list, alpha=0, beta=np.array([0, 1, 2]))
# shape (nL, nE, 3) → squeeze L
# P = P.squeeze(0)
print("P=", P)

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.figure(figsize=(6.5, 4.0))

plt.plot(Enu_list, P[:, 2], label=r"$P_{\mu\mu}$ disappearance", lw=2)
plt.plot(Enu_list, P[:, 3], label=r"$P_{\mu e}$ appearance", lw=2)
plt.plot(Enu_list, P.sum(axis=1), "--", label="Total probability", lw=1.5)

plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel(r"Probability")
plt.title(r"T2K-like vacuum oscillation ($L=295\,\mathrm{km}$)")
plt.xlim(E_min, E_max)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()
