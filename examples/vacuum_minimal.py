import numpy as np
from neutrino_interferometry.models.mixing import Mixing
from neutrino_interferometry.models.spectrum import Spectrum  # ta classe
from neutrino_interferometry.hamiltonian.base import Hamiltonian
import neutrino_interferometry.propagation.solvers as solvers

# 3 saveurs (exemple)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49.0)}
phases = {(1, 3): np.deg2rad(195)}

pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U = pmns.U()
print(np.round(U, 3))

spec = Spectrum(n=3, m_lightest=0.01)
spec.set_dm2({
    (2, 1): 7.42e-5,
    (3, 2): 0.0024428
})
spec.summary()
m2_diag = np.diag(spec.get_m2())

hamiltonian = Hamiltonian(U=U, m2_diag=m2_diag)

# Grille d’énergies et baseline
E = np.linspace(0.2, 5.0, 200)   # GeV
L = 295e3                        # m (T2K-like)

H_E = hamiltonian.vacuum(E)                # (nE,N,N)
S_E = solvers.propagator_vacuum(H_E, L)  # (nE,N,N)

# Probabilité ν_μ → ν_e
P_mue = solvers.probability_alpha_to_beta(S_E, alpha=2, beta=1)
print("P(νμ→νe) shape:", P_mue.shape)
