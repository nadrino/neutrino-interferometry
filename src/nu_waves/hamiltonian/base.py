from nu_waves.backends import make_numpy_backend
from nu_waves.utils.units import VCOEFF_EV
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend

from abc import ABC, abstractmethod


class Hamiltonian(ABC):
    def __init__(self, mixing_matrix, m2_array):
        xp = Backend().xp()
        self.mixing_matrix = xp.asarray(mixing_matrix, dtype=Backend().complex_dtype())
        self.mixing_matrix_dagger = xp.conjugate(self.mixing_matrix).T
        self.m2 = xp.asarray(m2_array, dtype=Backend().real_dtype())
        self.n_flavors = self.mixing_matrix.shape[0]

    # Default: produce S(L) in FLAVOR basis
    @abstractmethod
    def get_barger_propagator(self, L, E=None) -> any:
        """Return S(L) in FLAVOR basis, shape (nE, nF, nF)."""
        ...

    # Generic propagation (can be overridden for faster calculations)
    def propagate_state(self, psi: WaveFunction, L, E=None):
        # Ensure ψ is in a flavor basis for the default path
        # Psi will be returned expressed in the flavor basis
        if psi.current_basis != Basis.FLAVOR:
            raise ValueError("Default Hamiltonian expects psi in FLAVOR basis. "
                             "Override propagate_state in your subclass, or rotate ψ beforehand.")

        S = self.get_barger_propagator(L=L, E=E)                       # (nE,nF,nF), flavor basis
        psi.values = (S @ psi.values[..., None])[..., 0]    # (nE,nF)


# class Hamiltonian:
#     def __init__(self, mixing_matrix, m2_diag, backend=None):
#         """
#         U: (N,N) complex PMNS (ou 3+N)
#         m2_diag: (N,N) diag(m_i^2) [eV^2]
#         """
#         self.backend = None
#         self.set_backend(backend)
#
#         xp = self.backend.xp
#         self.U = xp.asarray(mixing_matrix, dtype=self.backend.dtype_complex)
#         m2 = xp.asarray(m2_diag, dtype=self.backend.dtype_real)
#         if m2.ndim == 2:
#             # accept diagonal matrix but store vector
#             m2 = xp.diag(m2)
#         self.m2_diag = m2.reshape(-1)
#
#         self.n_flavors = self.U.shape[0]
#
#         if self.U.shape[0] != self.m2_diag.shape[0]:
#             raise ValueError(f"Mixing matrix shape {mixing_matrix.shape} incompatible with m2_diag size {len(m2_diag)}")
#
#         # self.U = mixing_matrix
#         # self.m2_diag = m2_diag
#
#     def set_backend(self, backend):
#         self.backend = backend or make_numpy_backend()
#
#     def vacuum(self, E, antineutrino: bool = False):
#         """
#         Return the flavor-basis Hamiltonian in vacuum.
#
#         Parameters
#         ----------
#         E_GeV : float or array
#             Neutrino energy in GeV.
#         antineutrino : bool, optional
#             If True, uses complex-conjugated mixing matrix (U*).
#         """
#         xp = self.backend.xp
#         U = xp.conjugate(self.U) if antineutrino else self.U
#         H0 = (U * self.m2_diag[self.backend.xp.newaxis, :]) @ self.backend.xp.swapaxes(self.backend.xp.conj(U), -1, -2)
#         H = H0 / (2.0 * E[..., xp.newaxis, xp.newaxis])  # ()→(N,N) or (nE,N,N)
#         return H
#
#     def matter_constant(self,
#                         E,
#                         rho_gcm3: float,
#                         Ye: float = 0.5,
#                         antineutrino: bool = False):
#         """
#         H_matter(E) = U diag(m^2)/(2E) U^† + diag(Ve, 0, 0, ...),
#         with Ve = + 7.632e-14 * rho[g/cm^3] * Ye [eV] for neutrinos,
#                 = - Ve for antineutrinos.
#         Works for scalar or vector E and arbitrary dimension >= 1.
#         """
#         xp = self.backend.xp
#
#         # Vacuum term (respect anti-ν via U*)
#         U = xp.conj(self.U) if antineutrino else self.U
#         H0 = (U * self.m2_diag[xp.newaxis, :]) @ xp.swapaxes(xp.conj(U), -1, -2)  # (N,N)
#         H_vac = H0 / (2.0 * (E)[..., xp.newaxis, xp.newaxis])  # (..,N,N)
#
#         # Matter potential in flavor basis (complex dtype for uniformity)
#         N = self.U.shape[0]
#         V_f = xp.zeros((N, N), dtype=self.backend.dtype_complex)
#         Ve = xp.asarray(VCOEFF_EV * rho_gcm3 * Ye, dtype=self.backend.dtype_real)
#         sign = -1.0 if antineutrino else +1.0
#         V_f = V_f + 0  # no-op; safe to remove if you prefer
#         V_f[..., 0, 0] = sign * Ve  # only e-flavor gets the CC potential
#
#         # Broadcast across energies if vector E
#         return H_vac + (V_f if E.ndim == 0 else V_f[xp.newaxis, :, :])

