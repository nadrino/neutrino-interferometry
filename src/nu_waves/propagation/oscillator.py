import numpy as np
from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.propagation.solvers import KM_TO_EVINV


class VacuumOscillator:
    """
    Compute oscillation probabilities in vacuum for arbitrary (L, E) pairs or grids.

    Parameters
    ----------
    mixing_matrix : np.ndarray
        PMNS-like mixing matrix (N,N).
    m2_diag : np.ndarray
        Mass-squared values [eV^2].
    """

    def __init__(self, mixing_matrix: np.ndarray, m2_diag: np.ndarray):
        self.hamiltonian = Hamiltonian(mixing_matrix, m2_diag)

    # ----------------------------------------------------------------------

    def probability(self,
                    L_km: float | np.ndarray,
                    E_GeV: float | np.ndarray,
                    alpha: int | np.ndarray | None = None,
                    beta: int | np.ndarray | None = None) -> np.ndarray:
        """
        Compute P_{alpha -> beta}(L, E).

        Parameters
        ----------
        alpha, beta : int or array or None
            Flavor indices (0=e, 1=mu, 2=tau).
            - If both None: returns full (N,N) matrix for each (L,E)
            - If arrays: broadcasted selection of indices
        L_km : array or float
            Baseline(s) in km
        E_GeV : array or float
            Neutrino energy(ies) in GeV

        Returns
        -------
        P : ndarray
            If alpha,beta are None: shape (nL, nE, N, N)
            Else: shape broadcasted over L,E plus flavor indices.
        """
        L_km = np.atleast_1d(L_km).astype(float)
        E_GeV = np.atleast_1d(E_GeV).astype(float)
        nL, nE = L_km.size, E_GeV.size

        # Hamiltonian and diagonalization
        H = self.hamiltonian.vacuum(E_GeV)     # (nE, N, N)
        eigvals, eigvecs = np.linalg.eigh(H)   # (nE, N), (nE, N, N)
        N = eigvals.shape[1]

        # Build propagators for all (L,E)
        L_phase = (L_km[:, None, None] * KM_TO_EVINV)  # (nL,1,1)
        phase = np.exp(-1j * eigvals[None, :, :] * L_phase)  # (nL,nE,N)

        # Reconstruct S_{βα}(L,E)
        S = np.einsum("eik,lek->lei k", eigvecs, phase) @ np.conjugate(eigvecs[:, None, :, :])
        # Shape (nL,nE,N,N)

        P = np.abs(S) ** 2  # (nL,nE,N,N)

        # Handle flavor selection
        if alpha is None and beta is None:
            return P  # full tensor (nL,nE,N,N)

        alpha = np.atleast_1d(alpha)
        beta = np.atleast_1d(beta)

        P_sel = P[..., beta[:, None], alpha[None, :]]  # (nL,nE,len(beta),len(alpha))

        # Simplify shape if single indices
        if P_sel.shape[-2:] == (1, 1):
            return P_sel[..., 0, 0]
        return np.squeeze(P_sel)
