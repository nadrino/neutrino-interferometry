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

    def __init__(self,
                 mixing_matrix: np.ndarray,
                 m2_list: np.ndarray,
                 energy_sampler=None,
                 baseline_sampler=None,
                 n_samples=100
                 ):
        self.hamiltonian = Hamiltonian(mixing_matrix, m2_list)

        # samplers: callable(center_array, n_samples)
        self.energy_sampler = energy_sampler
        self.baseline_sampler = baseline_sampler
        self.n_samples = n_samples

    # ----------------------------------------------------------------------
    def probability(self,
                    alpha: int | np.ndarray | None = None,
                    beta:  int | np.ndarray | None = None,
                    L_km:  np.ndarray | float = 0.0,
                    E_GeV: np.ndarray | float = 1.0,
                    antineutrino: bool = False
                    ) -> np.ndarray:
        # ---------- normalize inputs & detect grid/pairs ----------
        L_in = np.asarray(L_km, float)
        E_in = np.asarray(E_GeV, float)

        grid_mode = (L_in.ndim == 1 and E_in.ndim == 1 and L_in.size > 1 and E_in.size > 1)
        if grid_mode:
            Lc, Ec = np.meshgrid(L_in, E_in, indexing="ij")          # (nL,nE)
        else:
            Lc, Ec = np.broadcast_arrays(L_in, E_in)                  # same-shape S
            if Lc.ndim == 0:  # both scalars
                Lc = Lc.reshape(1); Ec = Ec.reshape(1)

        center_shape = Lc.shape                                       # S

        # ---------- sampling or no-sampling paths ----------
        use_sampling = (self.energy_sampler is not None) or (self.baseline_sampler is not None)
        if not use_sampling:
            # --- original path (no overhead) ---
            E_flat = Ec.reshape(-1)                                   # (B,)
            L_flat = Lc.reshape(-1)                                   # (B,)
            H = self.hamiltonian.vacuum(E_flat, antineutrino=antineutrino)  # (B,N,N)
            eigvals, eigvecs = np.linalg.eigh(H)                            # (B,N),(B,N,N)
            phase = np.exp(-1j * eigvals * (L_flat * KM_TO_EVINV)[:, None]) # (B,N)
            S = np.einsum("bik,bk,bjk->bij", eigvecs, phase, eigvecs.conj()) # (B,N,N)
            P = (np.abs(S) ** 2).reshape(*center_shape, S.shape[-2], S.shape[-1]) # S+(N,N)
        else:
            # --- smeared path ---
            ns = int(max(1, self.n_samples))

            def _tile(x):
                return np.repeat(x[..., None], ns, axis=-1)

            Es = self.energy_sampler(Ec, ns) if self.energy_sampler else _tile(Ec)  # S+(ns,)
            Ls = self.baseline_sampler(Lc, ns) if self.baseline_sampler else _tile(Lc)

            E_flat = Es.reshape(-1)
            L_flat = Ls.reshape(-1)

            H = self.hamiltonian.vacuum(E_flat, antineutrino=antineutrino)  # (S*ns,N,N)
            eigvals, eigvecs = np.linalg.eigh(H)
            phase = np.exp(-1j * eigvals * (L_flat * KM_TO_EVINV)[:, None])
            S = np.einsum("bik,bk,bjk->bij", eigvecs, phase, eigvecs.conj())       # (S*ns,N,N)
            P = (np.abs(S) ** 2).reshape(*center_shape, ns, S.shape[-2], S.shape[-1]).mean(axis=-3)  # S+(N,N)

        # ---------- squeeze scalar axes like before ----------
        if not grid_mode:
            if L_in.ndim == 0 and E_in.ndim == 0:     # both scalars
                P = P[0]
            elif P.shape[0] == 1:
                P = P[0]

        # ---------- flavor selection (same rules as before) ----------
        def _as_idx(x, N):
            if x is None:
                return np.arange(N)
            x = np.asarray(x)
            return int(x) if x.ndim == 0 else x

        N = P.shape[-1]
        a = _as_idx(alpha, N)
        b = _as_idx(beta,  N)

        if alpha is None and beta is None:
            return P

        a_scalar = np.isscalar(a)
        b_scalar = np.isscalar(b)

        if a_scalar and b_scalar:
            return P[..., b, a]
        if a_scalar and not b_scalar:
            return P[..., b, a]
        if not a_scalar and b_scalar:
            return P[..., b, a]
        return P[..., np.ix_(b, a)]

