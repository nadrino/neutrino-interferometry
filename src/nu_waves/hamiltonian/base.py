from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.models.spectrum import Spectrum
from nu_waves.models.mixing import Mixing

from abc import ABC, abstractmethod


class HamiltonianBase(ABC):
    def __init__(self, mixing: Mixing, spectrum: Spectrum, antineutrino: bool):
        self._antineutrino = antineutrino
        self._mixing = mixing
        self._spectrum = spectrum
        self._check_parameters()

    @property
    def n_flavors(self):
        return self._mixing.n_neutrinos

    @property
    def mixing(self):
        return self._mixing

    @property
    def spectrum(self):
        return self._spectrum

    def set_spectrum(self, spectrum: Spectrum):
        self._spectrum = spectrum
        self._check_parameters()

    # Default: produce S(L) in FLAVOR basis
    @abstractmethod
    def get_barger_propagator(self, L, E=None) -> any:
        """Return S(L) in FLAVOR basis, shape (nE, nF, nF)."""
        ...

    # Generic propagation (can be overridden for faster calculations)
    # psi returned should be expressed in the flavor basis
    def propagate_state(self, psi: WaveFunction, L, E=None):
        # Ensure ψ is in a flavor basis for the default path
        if psi.current_basis != Basis.FLAVOR:
            raise ValueError("Default Hamiltonian expects psi in FLAVOR basis. "
                             "Override propagate_state in your subclass, or rotate ψ beforehand.")

        S = self.get_barger_propagator(L=L, E=E)                       # (nE,nF,nF), flavor basis
        psi.values = (S @ psi.values[..., None])[..., 0]    # (nE,nF)

    def _check_parameters(self):
        assert (self._spectrum.n_neutrinos == self._mixing.n_neutrinos)