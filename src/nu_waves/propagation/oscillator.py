import numpy as np
from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.backends import make_numpy_backend
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
                 n_samples=100,
                 backend=None
                 ):
        self.backend = backend or make_numpy_backend()
        self.hamiltonian = Hamiltonian(
            mixing_matrix, m2_list,
            backend=self.backend
        )

        # samplers: callable(center_array, n_samples)
        self.energy_sampler = energy_sampler
        self.baseline_sampler = baseline_sampler
        self.n_samples = n_samples

    # ----------------------------------------------------------------------
    def probability(self,
                    alpha = None,
                    beta = None,
                    L_km = 0.0,
                    E_GeV = 1.0,
                    antineutrino: bool = False
                    ):
        xp = self.backend.xp
        linalg = self.backend.linalg

        # ---------- normalize inputs & detect grid/pairs ----------
        L_in = xp.asarray(L_km, dtype=self.backend.dtype_real)
        E_in = xp.asarray(E_GeV, dtype=self.backend.dtype_real)

        grid_mode = (L_in.ndim == 1 and E_in.ndim == 1 and L_in.size > 1 and E_in.size > 1)
        if grid_mode:
            Lc, Ec = xp.meshgrid(L_in, E_in, indexing="ij")          # (nL,nE)
        else:
            Lc, Ec = xp.broadcast_arrays(L_in, E_in)                  # same-shape S
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
            eigvals, eigvecs = linalg.eigh(H)                            # (B,N),(B,N,N)
            KM = xp.asarray(KM_TO_EVINV, dtype=self.backend.dtype_real)
            phase = xp.exp(-1j * eigvals * (L_flat * KM)[:, None]) # (B,N)
            S = xp.einsum("bik,bk,bjk->bij", eigvecs, phase, eigvecs.conj()) # (B,N,N)
            P = (xp.abs(S) ** 2).reshape(*center_shape, S.shape[-2], S.shape[-1]) # S+(N,N)
        else:
            # --- smeared path ---
            ns = int(max(1, self.n_samples))

            # def _tile(x):
            #     return xp.repeat_last(x, ns)  # instead of np.repeat(..., axis=-1)
            #
            def _tile(x):
                return xp.repeat_last(x, ns)

            Es = self.energy_sampler(Ec, ns) if self.energy_sampler else _tile(Ec)  # S+(ns,)
            Ls = self.baseline_sampler(Lc, ns) if self.baseline_sampler else _tile(Lc)

            E_flat = Es.reshape(-1)
            L_flat = Ls.reshape(-1)

            H = self.hamiltonian.vacuum(E_flat, antineutrino=antineutrino)  # (S*ns,N,N)
            eigvals, eigvecs = xp.linalg.eigh(H)
            KM = xp.asarray(KM_TO_EVINV, dtype=self.backend.dtype_real)
            phase = xp.exp(-1j * eigvals * (L_flat * KM)[:, None])
            S = xp.einsum("bik,bk,bjk->bij", eigvecs, phase, eigvecs.conj())       # (S*ns,N,N)
            P = (xp.abs(S) ** 2).reshape(*center_shape, ns, S.shape[-2], S.shape[-1]).mean(axis=-3)  # S+(N,N)

        # ---------- squeeze scalar axes like before ----------
        if not grid_mode:
            if L_in.ndim == 0 and E_in.ndim == 0:     # both scalars
                P = P[0]
            elif P.shape[0] == 1:
                P = P[0]

        # ---------- flavor selection (same rules as before) ----------
        def _as_idx(x, N):
            if x is None:
                return xp.arange(N)
            x = xp.asarray(x)
            return int(x) if x.ndim == 0 else x

        N = P.shape[-1]
        a = _as_idx(alpha, N)
        b = _as_idx(beta,  N)

        is_torch = hasattr(self.backend.xp, "device") and str(type(self.backend.xp)).startswith(
            "<class 'nu_waves.backends.torch_backend._TorchXP'")

        if alpha is None and beta is None:
            return P

        a_scalar = xp.isscalar(a)
        b_scalar = xp.isscalar(b)

        if not is_torch:
            # NumPy path (unchanged)
            if a_scalar and b_scalar:       return P[..., b, a]
            if a_scalar and not b_scalar:   return P[..., b, a]
            if not a_scalar and b_scalar:   return P[..., b, a]
            return P[..., self.backend.xp.ix_(b, a)]
        else:
            # Torch path uses index_select (advanced indexing parity)
            import torch
            to_idx = lambda x: x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.long,
                                                                                     device=self.backend.xp.device)
            if a_scalar and b_scalar:
                return P[..., int(b), int(a)]
            if a_scalar and not b_scalar:
                return P.index_select(-2, to_idx(b))[..., int(a)]
            if not a_scalar and b_scalar:
                return P.index_select(-1, to_idx(a)).index_select(-2, to_idx(b))
            # both arrays
            P_sel = P.index_select(-2, to_idx(b)).index_select(-1, to_idx(a))  # (..., len(b), len(a))
            return P_sel

