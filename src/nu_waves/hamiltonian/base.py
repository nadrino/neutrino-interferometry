from nu_waves.backends import make_numpy_backend
from nu_waves.utils.units import GEV_TO_EV
from nu_waves.utils.units import VCOEFF_EV


class Hamiltonian:
    def __init__(self, mixing_matrix, m2_diag, backend=None):
        """
        U: (N,N) complex PMNS (ou 3+N)
        m2_diag: (N,N) diag(m_i^2) [eV^2]
        """
        self.backend = backend or make_numpy_backend()
        xp = self.backend.xp

        self.U = xp.asarray(mixing_matrix, dtype=self.backend.dtype_complex)
        m2 = xp.asarray(m2_diag, dtype=self.backend.dtype_real)
        if m2.ndim == 2:
            # accept diagonal matrix but store vector
            m2 = xp.diag(m2)
        self.m2_diag = m2.reshape(-1)
        # self.U = mixing_matrix
        # self.m2_diag = m2_diag

    def vacuum(self, E_GeV, antineutrino: bool = False):
        """
        Return the flavor-basis Hamiltonian in vacuum.

        Parameters
        ----------
        E_GeV : float or array
            Neutrino energy in GeV.
        antineutrino : bool, optional
            If True, uses complex-conjugated mixing matrix (U*).
        """
        xp = self.backend.xp
        E = xp.asarray(E_GeV, dtype=self.backend.dtype_real)
        E = E.reshape(()) if E.ndim == 0 else E.reshape(-1)  # () or (nE,)

        EV_PER_GEV = self.backend.xp.asarray(GEV_TO_EV, dtype=self.backend.dtype_real)
        E_eV = E * EV_PER_GEV
        # E_eV = E * GEV_TO_EV

        U = xp.conjugate(self.U) if antineutrino else self.U
        # (U * m2) @ U^†  (scale columns by m2)
        # H0 = (U * self.m2_diag[xp.newaxis, :]) @ xp.conjugate(U).T  # (N,N)
        # H0 = (U * self.m2_diag[xp.newaxis, :]) @ xp.swapaxes(xp.conj(U), -1, -2)
        H0 = (U * self.m2_diag[self.backend.xp.newaxis, :]) @ self.backend.xp.swapaxes(self.backend.xp.conj(U), -1, -2)
        H = H0 / (2.0 * E_eV[..., xp.newaxis, xp.newaxis])  # ()→(N,N) or (nE,N,N)
        return H

        # E_eV = np.asarray(E_GeV, dtype=float) * GEV_TO_EV
        # U = np.conjugate(self.U) if antineutrino else self.U
        # D = np.diag(self.m2_diag)
        # H = U @ D @ U.conj().T
        # H = H / (2.0 * E_eV[..., None, None])  # broadcast over E
        # return H

    def matter_constant(self,
                        E_GeV,
                        rho_gcm3: float,
                        Ye: float = 0.5,
                        antineutrino: bool = False):
        """
        H_matter(E) = U diag(m^2)/(2E) U^† + diag(Ve, 0, 0, ...),
        with Ve = + 7.632e-14 * rho[g/cm^3] * Ye [eV] for neutrinos,
                = - Ve for antineutrinos.
        Works for scalar or vector E and arbitrary dimension >= 1.
        """
        xp = self.backend.xp

        # Energies on backend dtype/device
        E = xp.asarray(E_GeV, dtype=self.backend.dtype_real)
        E = E.reshape(()) if E.ndim == 0 else E.reshape(-1)  # () or (nE,)
        EV_PER_GEV = xp.asarray(GEV_TO_EV, dtype=self.backend.dtype_real)

        # Vacuum term (respect anti-ν via U*)
        U = xp.conj(self.U) if antineutrino else self.U
        H0 = (U * self.m2_diag[xp.newaxis, :]) @ xp.swapaxes(xp.conj(U), -1, -2)  # (N,N)
        H_vac = H0 / (2.0 * (E * EV_PER_GEV)[..., xp.newaxis, xp.newaxis])  # (..,N,N)

        # Matter potential in flavor basis (complex dtype for uniformity)
        N = self.U.shape[0]
        V_f = xp.zeros((N, N), dtype=self.backend.dtype_complex)
        Ve = xp.asarray(VCOEFF_EV * rho_gcm3 * Ye, dtype=self.backend.dtype_real)
        sign = -1.0 if antineutrino else +1.0
        V_f = V_f + 0  # no-op; safe to remove if you prefer
        V_f[..., 0, 0] = sign * Ve  # only e-flavor gets the CC potential

        # Broadcast across energies if vector E
        return H_vac + (V_f if E.ndim == 0 else V_f[xp.newaxis, :, :])

