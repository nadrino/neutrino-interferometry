from nu_waves.hamiltonian.base import HamiltonianBase
from nu_waves.models.spectrum import Spectrum
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.models.mixing import Mixing
from nu_waves.utils.units import VCOEFF_EV


class Hamiltonian(HamiltonianBase):
    """
        Matter Hamiltonian with Barger layer product.


    """
    def __init__(self, mixing: Mixing, spectrum: Spectrum, antineutrino: bool):
        super().__init__(mixing=mixing, spectrum=spectrum, antineutrino=antineutrino)
        self._constant_profile = None
        self.set_constant_density(rho_in_g_per_cm3=0)

    def set_constant_density(self, rho_in_g_per_cm3: float, Ye: float = 0.5):
        self._constant_profile = (rho_in_g_per_cm3, Ye)

    def get_barger_propagator(self, L, E):
        xp = Backend().xp()

        # make sure they are matching the backend (no copy if already matching)
        E = xp.asarray(E)
        L = xp.asarray(L)

        U = self._mixing.build_mixing_matrix()
        Ud = xp.conjugate(U.T)
        m2 = xp.asarray(self.spectrum.get_m2(), dtype=U.dtype)
        H_vacuum_eV2 = U @ xp.diag(m2) @ Ud

        flavor_projector = xp.zeros((self.n_neutrinos, self.n_neutrinos), dtype=U.dtype)
        flavor_projector[0, 0] = 1.0  # only electron neutrinos feel the potential

        rho, Ye = self._constant_profile
        signA = -1.0 if self._antineutrino else 1.0
        matter_potential = signA * (VCOEFF_EV * rho * Ye)

        inv2E = (0.5 / E)[:, None, None]
        H = H_vacuum_eV2[None, :, :] * inv2E + matter_potential * flavor_projector[None, :, :]

        eigen_values, eigen_vectors = xp.linalg.eigh(H)
        phases = xp.exp(-1j * eigen_values * L[:, None])
        S = (eigen_vectors * phases[:, None, :]) @ xp.conjugate(eigen_vectors).mT
        return S

    # def set_layers(self, L_layers_km, rho_layers_gcc, Ye_layers = None):
    #     """Set a piecewise-constant profile (Barger 'castle-wall')."""
    #     xp = Backend().xp()
    #     Ls = xp.asarray(L_layers_km, dtype=Backend().real_dtype())
    #     rhos = xp.asarray(rho_layers_gcc, dtype=Backend().real_dtype())
    #     if Ye_layers is None:
    #         Yes = xp.full_like(rhos, 0.5, dtype=Backend().real_dtype())
    #     else:
    #         Yes = xp.asarray(Ye_layers, dtype=Backend().real_dtype())
    #     if not (Ls.shape == rhos.shape == Yes.shape):
    #         raise ValueError("Layers L, rho, Ye must have the same shape.")
    #     self._layers = (Ls, rhos, Yes)




