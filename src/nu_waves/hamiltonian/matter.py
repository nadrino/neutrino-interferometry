from nu_waves.hamiltonian.base import HamiltonianBase
from nu_waves.models.spectrum import Spectrum
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.models.mixing import Mixing


class Hamiltonian(HamiltonianBase):
    def __init__(self, mixing: Mixing, spectrum: Spectrum, antineutrino: bool):
        super().__init__(mixing=mixing, spectrum=spectrum, antineutrino=antineutrino)

    def get_barger_propagator(self, L, E=None):
        xp = Backend().xp()

        S = ... # do the calculations
        return S

