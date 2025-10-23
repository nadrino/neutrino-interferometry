from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend


class VacuumHamiltonian(Hamiltonian):
    def __init__(self, mixing_matrix, m2_array):
        super().__init__(mixing_matrix=mixing_matrix, m2_array=m2_array)

    def propagate_state(self, psi: WaveFunction, L, E=None):
        xp = Backend().xp()

        # Rotate psi to mass basis if needed
        if psi.current_basis == Basis.FLAVOR:
            psi.to_basis(Basis.MASS, self.mixing_matrix_dagger)

        if psi.current_basis != Basis.MASS:
            raise ValueError(f"VacuumHamiltonian: unsupported input basis for psi: {psi.current_basis}")

        # phases φ_i = 1.267 * Δm_i^2[eV^2] * L[eV-1] / E[eV]
        phases = 0.5 * (L / E)[:, None] * self.m2[None, :]    # (nE, nF)
        D = xp.exp(-1j * phases)[:, None, :]                    # (nE, 1, nF)

        # Diagonal propagation in mass basis
        psi.values = psi.values * D                             # (nE, nFe, nF)

        # convention
        psi.to_basis(Basis.FLAVOR, self.mixing_matrix)

    def get_barger_propagator(self, L, E=None):
        xp = Backend().xp()

        phases = 1.267 * (L / E)[:, None] * self.m2[None, :]
        D = xp.exp(-1j * phases)

        S = (self.mixing_matrix[None, :, :] * D[:, None, :]) @ self.mixing_matrix_dagger[None, :, :]
        return S

