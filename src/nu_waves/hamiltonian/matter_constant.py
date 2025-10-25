from nu_waves.hamiltonian.base import HamiltonianBase
from nu_waves.models.spectrum import Spectrum
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.models.mixing import Mixing
from nu_waves.utils.units import VCOEFF_EV, KM_TO_EVINV

from dataclasses import dataclass


@dataclass
class MatterLayer:
    rho_in_g_per_cm3: float     # matter density
    Ye: float                   # avg number of electrons per nucleon (0.5 by default)
    weight: float               # either absolute length [eV^-1] or fraction [0..1]


@dataclass
class MatterProfile:
    layers: list[MatterLayer]
    slicing: str  # "fraction" or "absolute"

    @staticmethod
    def from_fractions(rho_gcm3, Ye, fractions) -> "MatterProfile":
        fr = Backend.xp().asarray(fractions, float)
        Ls = [MatterLayer(r, y, w) for r, y, w in zip(rho_gcm3, Ye, fr)]
        return MatterProfile(Ls, "fraction")

    @staticmethod
    def from_segments(rho_gcm3, Ye, lengths_km) -> "MatterProfile":
        Ls = [MatterLayer(r, y, L * KM_TO_EVINV) for r, y, L in zip(rho_gcm3, Ye, lengths_km)]
        return MatterProfile(Ls, "absolute")

    def resolve_dL(self, L_in_eV_inv) -> list[Backend.xp().ndarray]:
        """
        Map total baselines (array-like) to per-layer ΔL_k arrays.
        Returns list of arrays, one per layer, each shaped like L_total_km_array.
        - fraction layers scale with L_total
        - absolute layers ignore L_total (assumed consistent with physics setup)
        """
        Ltot = Backend.xp().asarray(L_in_eV_inv, float)
        dLs = list()
        if self.slicing == "fraction":
            for layer in self.layers:
                dLs.append(layer.weight * Ltot)
        elif self.slicing == "absolute":
            for layer in self.layers:
                dLs.append(Backend.xp().full_like(Ltot, layer.weight, dtype=float))
        else:
            raise NotImplementedError("slicing must be 'fraction' or 'absolute'")

        return dLs


class Hamiltonian(HamiltonianBase):
    """
        Matter Hamiltonian with Barger layer product.


    """
    def __init__(self, mixing: Mixing, spectrum: Spectrum, antineutrino: bool):
        super().__init__(mixing=mixing, spectrum=spectrum, antineutrino=antineutrino)
        self._constant_profile = None
        self._matter_profile = None
        self.set_constant_density(rho_in_g_per_cm3=0)

    def set_constant_density(self, rho_in_g_per_cm3: float, Ye: float = 0.5):
        self._constant_profile = (rho_in_g_per_cm3, Ye)

    def set_matter_profile(self, matter_profile: MatterProfile):
        self._matter_profile = matter_profile

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

        signA = -1.0 if self._antineutrino else 1.0
        inv2E = (0.5 / E)[:, None, None] # used for the hamiltonian

        if self._matter_profile is not None:
            L_slices = self._matter_profile.resolve_dL(L)
            # Create a (1, n_nu, n_nu) identity matrix in complex dtype for initialization
            S = xp.eye(self.n_neutrinos, dtype=Backend.complex_dtype())[None, ...]
            # Broadcast the identity across all energies and make it writable (hence the .copy)
            S = xp.copy(xp.broadcast_to(S, (E.shape[0], self.n_neutrinos, self.n_neutrinos)))
            for k, layer in enumerate(self._matter_profile.layers):
                matter_potential = signA * (VCOEFF_EV * layer.rho_in_g_per_cm3 * layer.Ye)
                H = H_vacuum_eV2[None, ...] * inv2E + matter_potential * flavor_projector[None, ...]
                eigen_values, eigen_vectors = xp.linalg.eigh(H)
                phases = xp.exp((-1j) * eigen_values * L_slices[k][..., None])
                Sk = (eigen_vectors * phases[:, None, :]) @ xp.conjugate(eigen_vectors).mT
                S = Sk @ S
        else:
            rho, Ye = self._constant_profile
            matter_potential = signA * (VCOEFF_EV * rho * Ye)
            H = H_vacuum_eV2[None, ...] * inv2E + matter_potential * flavor_projector[None, ...]

            eigen_values, eigen_vectors = xp.linalg.eigh(H)
            phases = xp.exp(-1j * eigen_values * L[..., None])
            S = (eigen_vectors * phases[:, None, :]) @ xp.conjugate(eigen_vectors).mT

        return S
