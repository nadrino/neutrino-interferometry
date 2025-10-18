import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian.base import Hamiltonian
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

Enu_list = np.linspace(0.2, 3.0, 200)
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
