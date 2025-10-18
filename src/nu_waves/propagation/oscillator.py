import numpy as np
from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.utils.units import KM_TO_EVINV


class VacuumOscillator:
    """
    Compute oscillation probabilities in vacuum for arbitrary (L, E) pairs or grids.

    Parameters
    ----------
    mixing_matrix : np.ndarray
        PMNS-like mixing matrix (N,N).
    m2_list : np.ndarray
        Mass-squared values [eV^2].
    """

    def __init__(self, mixing_matrix: np.ndarray, m2_list: np.ndarray):
        self.hamiltonian = Hamiltonian(mixing_matrix, m2_list)

    # ----------------------------------------------------------------------

    def probability(self,
                    L_km: float | np.ndarray,
                    E_GeV: float | np.ndarray,
                    alpha: int | np.ndarray | None = None,
                    beta: int | np.ndarray | None = None,
                    antineutrino: bool = False
                    ) -> np.ndarray:
        """
        Compute P_{alpha -> beta}(L, E).

        Parameters
        ----------
        L_km : array or float
            Baseline(s) in km
        E_GeV: array or float
            Neutrino energy(ies) in GeV
        alpha, beta : int or array or None
            Flavor indices (0=e, 1=mu, 2=tau).
            - If both None: returns full (N,N) matrix for each (L,E)
            - If arrays: broadcasted selection of indices
        antineutrino :
            Use the hamiltonian conjugate

        Returns
        -------
        P : ndarray
            If alpha,beta are None: shape (max(nE,nL), N, N)
            :param antineutrino:
        """
        L_km = np.atleast_1d(L_km).astype(float)
        E_GeV = np.atleast_1d(E_GeV).astype(float)
        nL, nE = L_km.size, E_GeV.size

        # Hamiltonian and diagonalization
        H = self.hamiltonian.vacuum(E_GeV=E_GeV, antineutrino=antineutrino)     # (nE, N, N)
        eigvals, eigvecs = np.linalg.eigh(H)   # (nE, N), (nE, N, N)
        N = eigvals.shape[1]

        # Build propagators for all (L,E)
        L_phase = (L_km[:, None, None] * KM_TO_EVINV)  # (nL,1,1)
        phase = np.exp(-1j * eigvals[None, :, :] * L_phase)  # (nL,nE,N)

        # Reconstruct S_{βα}(L,E)
        S = np.einsum("eik,lek->lei k", eigvecs, phase) @ np.conjugate(eigvecs[:, None, :, :])
        # Shape (nL,nE,N,N)

        # --- up to here you have P of shape (nL, nE, N, N) ---
        P = np.abs(S) ** 2  # (nL, nE, N, N)

        # Remember whether inputs were scalars
        L_is_scalar = np.ndim(L_km) == 0 or (np.ndim(L_km) == 1 and np.size(L_km) == 1)
        E_is_scalar = np.ndim(E_GeV) == 0 or (np.ndim(E_GeV) == 1 and np.size(E_GeV) == 1)

        # Squeeze the axes that were scalar in the inputs
        if L_is_scalar and E_is_scalar:
            P = P[0, 0]  # -> (N, N)
        elif L_is_scalar:
            P = P[0]  # -> (nE, N, N)
        elif E_is_scalar:
            P = P[:, 0]  # -> (nL, N, N)

        # else keep (nL, nE, N, N)

        # ---- Robust flavor selection (keeps expected shapes) ----
        def _as_idx(x, N):
            if x is None:
                return np.arange(N)
            x = np.asarray(x)
            return int(x) if x.ndim == 0 else x

        N = P.shape[-1]  # trailing axes are (beta, alpha)
        a = _as_idx(alpha, N)
        b = _as_idx(beta, N)

        if alpha is None and beta is None:
            return P  # (nE,N,N), (nL,N,N), (N,N), or (nL,nE,N,N)

        a_scalar = np.isscalar(a)
        b_scalar = np.isscalar(b)

        if a_scalar and b_scalar:
            return P[..., b, a]  # -> (nE), (nL), or scalar
        if a_scalar and not b_scalar:
            return P[..., b, a]  # -> (nE, len(beta)) or (nL, len(beta))
        if not a_scalar and b_scalar:
            return P[..., b, a]  # -> (nE, len(alpha)) or (nL, len(alpha))

        # both arrays → Cartesian selection
        return P[..., np.ix_(b, a)]  # -> (..., len(beta), len(alpha))

