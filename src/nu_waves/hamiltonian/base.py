import numpy as np
from nu_waves.utils.units import GEV_TO_EV

class Hamiltonian:
    def __init__(self, mixing_matrix: np.ndarray, m2_diag: np.ndarray):
        """
        U: (N,N) complex PMNS (ou 3+N)
        m2_diag: (N,N) diag(m_i^2) [eV^2]
        """
        self.U = mixing_matrix
        self.m2_diag = m2_diag

    def vacuum(self, E_GeV: np.ndarray | float, antineutrino: bool = False) -> np.ndarray:
        """
        Return the flavor-basis Hamiltonian in vacuum.

        Parameters
        ----------
        E_GeV : float or array
            Neutrino energy in GeV.
        antineutrino : bool, optional
            If True, uses complex-conjugated mixing matrix (U*).
        """
        E_eV = np.asarray(E_GeV, dtype=float) * GEV_TO_EV
        U = np.conjugate(self.U) if antineutrino else self.U
        D = np.diag(self.m2_diag)
        H = U @ D @ U.conj().T
        H = H / (2.0 * E_eV[..., None, None])  # broadcast over E
        return H
