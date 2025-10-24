from nu_waves.hamiltonian.base import HamiltonianBase
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend


class Hamiltonian(HamiltonianBase):
    def __init__(self, mixing: Mixing, spectrum: Spectrum, antineutrino: bool):
        super().__init__(mixing=mixing, spectrum=spectrum, antineutrino=antineutrino)

    def propagate_state(self, psi: WaveFunction, L, E=None):
        xp = Backend().xp()

        U = self._mixing.build_mixing_matrix()
        Ud = xp.conjugate(U.T)

        if self._antineutrino:
            U = xp.conjugate(U)
            Ud = xp.conjugate(Ud)

        # Rotate psi to mass basis if needed
        if psi.current_basis == Basis.FLAVOR:
            psi.to_basis(
                target_basis=Basis.MASS,
                eigen_vectors_dagger=Ud
            )

        if psi.current_basis != Basis.MASS:
            raise ValueError(f"VacuumHamiltonian: unsupported input basis for psi: {psi.current_basis}")

        # phases φ_i = 1/2 * Δm_i^2[eV^2] * L[eV-1] / E[eV]
        phases = 0.5 * (L / E)[:, None] * self._spectrum.get_m2()[None, :]     # (nE, nF)
        D = xp.exp(-1j * phases)[:, None, :]                    # (nE, 1, nF)

        # Diagonal propagation in mass basis
        psi.values = psi.values * D                             # (nE, nFe, nF)

        # convention
        psi.to_basis(
            target_basis=Basis.FLAVOR,
            eigen_vectors_dagger=U
        )

    def get_barger_propagator(self, L, E=None):
        xp = Backend().xp()

        U = self._mixing.build_mixing_matrix()
        Ud = xp.conjugate(U.T)

        if self._antineutrino:
            U = xp.conjugate(U)
            Ud = xp.conjugate(Ud)

        phases = 1.267 * (L / E)[:, None] * self._spectrum.get_m2()[None, :]
        D = xp.exp(-1j * phases)

        S = (U[None, :, :] * D[:, None, :]) @ Ud[None, :, :]
        return S

