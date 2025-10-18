import numpy as np

GEV_TO_EV = 1.0e9  # eV / GeV

class Hamiltonian:
    def __init__(self, mixing_matrix: np.ndarray, m2_diag: np.ndarray):
        """
        U: (N,N) complex PMNS (ou 3+N)
        m2_diag: (N,N) diag(m_i^2) [eV^2]
        """
        self.U = mixing_matrix
        self.m2_diag = m2_diag

    def vacuum(self, E: float | np.ndarray) -> np.ndarray:
        """
        H_f(E) = U diag(m^2)/(2E) U^†
        E: scalar or array (GeV); returns (N,N) if scalar, (nE,N,N) otherwise.
        """
        E = np.asarray(E, dtype=float)
        D = self.m2_diag / (2.0 * E[..., None, None] * GEV_TO_EV)  # broadcast sur E
        return self.U @ D @ self.U.conj().T

