import numpy as np
from nu_waves.backends import make_numpy_backend
from nu_waves.utils.units import GEV_TO_EV


class Hamiltonian:
    def __init__(self, mixing_matrix: np.ndarray, m2_diag: np.ndarray, backend=None):
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

    def vacuum(self, E_GeV, antineutrino: bool = False) -> np.ndarray:
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
        E_eV = E * GEV_TO_EV
        U = xp.conjugate(self.U) if antineutrino else self.U
        # (U * m2) @ U^†  (scale columns by m2)
        # H0 = (U * self.m2_diag[xp.newaxis, :]) @ xp.conjugate(U).T  # (N,N)
        H0 = (U * self.m2_diag[xp.newaxis, :]) @ xp.swapaxes(xp.conj(U), -1, -2)
        H = H0 / (2.0 * E_eV[..., xp.newaxis, xp.newaxis])  # ()→(N,N) or (nE,N,N)
        return H

        # E_eV = np.asarray(E_GeV, dtype=float) * GEV_TO_EV
        # U = np.conjugate(self.U) if antineutrino else self.U
        # D = np.diag(self.m2_diag)
        # H = U @ D @ U.conj().T
        # H = H / (2.0 * E_eV[..., None, None])  # broadcast over E
        # return H
