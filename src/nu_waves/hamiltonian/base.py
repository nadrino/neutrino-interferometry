import numpy as np


class Hamiltonian:
    def __init__(self, U: np.ndarray, m2_diag: np.ndarray):
        """
        U: (N,N) complex PMNS (ou 3+N)
        m2_diag: (N,N) diag(m_i^2) [eV^2]
        """
        self.U = U
        self.m2_diag = m2_diag

    def vacuum(self, E: float | np.ndarray) -> np.ndarray:
        """
        H_f(E) = U diag(m^2)/(2E) U^†
        E: scalar or array (GeV); returns (N,N) if scalar, (nE,N,N) otherwise.
        """
        E = np.asarray(E, dtype=float)
        D = self.m2_diag / (2.0 * E[..., None, None])  # broadcast sur E
        return self.U @ D @ self.U.conj().T

