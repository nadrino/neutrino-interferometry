import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian.base import Hamiltonian
import nu_waves.propagation.solvers as solvers


# sterile test
angles = {(1, 2): np.deg2rad(45)}
pmns = Mixing(dim=2, mixing_angles=angles)
U_pmns = pmns.get_mixing_matrix()
print(np.round(U_pmns, 3))


# 1 eV^2
spec = Spectrum(n=2, m_lightest=0.)
spec.set_dm2({(2, 1): 1})
spec.summary()
m2_diag = np.diag(spec.get_m2())

hamiltonian = Hamiltonian(mixing_matrix=U_pmns, m2_diag=m2_diag)
P_dis_test = solvers.probability_alpha_to_beta(
    solvers.propagator_vacuum(
        hamiltonian.vacuum(1),
        10
    ),
    alpha=1, beta=1
)
print("P(survival nu_e)", P_dis_test)

E_list = np.linspace(0.5, 20, 200)
P_dis = solvers.probability_alpha_to_beta(
    solvers.propagator_vacuum(
        hamiltonian.vacuum(E_list),
        10
    ),
    alpha=1, beta=1
)
plt.figure(figsize=(6,4))
plt.plot(E_list, P_dis, lw=2)
plt.xlabel(r"$E_\nu$ [MeV]")
plt.ylabel(r"$P(\nu_e \to \nu_e)$")
plt.title("Vacuum oscillation probability probability")
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.show()
