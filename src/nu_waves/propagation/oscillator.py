import numpy as np
from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.backends import make_numpy_backend
from nu_waves.matter.profile import MatterProfile
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

    def set_constant_density(self, rho_gcm3: float, Ye: float = 0.5):
        self._use_matter = True
        self._matter_args = (float(rho_gcm3), float(Ye))

    def set_layered_profile(self, profile: MatterProfile):
        self._use_matter = True
        self._matter_profile = profile

    def use_vacuum(self):
        self._use_matter = False
        self._matter_args = None
        self._matter_profile = None

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
        else:
            # --- smeared path ---
            ns = int(max(1, self.n_samples))
            def _tile(x): return xp.repeat_last(x, ns)
            Es = self.energy_sampler(Ec, ns) if self.energy_sampler else _tile(Ec)  # S+(ns,)
            Ls = self.baseline_sampler(Lc, ns) if self.baseline_sampler else _tile(Lc)
            E_flat = Es.reshape(-1)
            L_flat = Ls.reshape(-1)


        KM = xp.asarray(KM_TO_EVINV, dtype=self.backend.dtype_real)

        if getattr(self, "_use_matter", False):

            if getattr(self, "_matter_profile", None) is None:
                # constant matter density
                rho, Ye = self._matter_args  # set via helper
                H = self.hamiltonian.matter_constant(E_flat, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)
                HL = H * (L_flat * KM)[:, xp.newaxis, xp.newaxis]
                S = linalg.matrix_exp((-1j) * HL)
            else:
                # layered profile
                prof = self._matter_profile
                # per-center ΔL_k arrays, each shaped like L_flat (broadcast-safe across grid or pairs)
                dL_list = prof.resolve_dL(self.backend.from_device(L_flat))  # resolve in host float
                dL_list = [xp.asarray(dL, dtype=self.backend.dtype_real) for dL in dL_list]

                # accumulate S_tot = S_K @ ... @ S_1 (source→detector order = list order)
                N = self.hamiltonian.U.shape[0]
                S = xp.eye(N, dtype=self.backend.dtype_complex)[xp.newaxis, ...]  # (1,N,N) seed
                S = xp.broadcast_to(S, (E_flat.shape[0], N, N)).copy()  # (B,N,N) identities

                for k, layer in enumerate(prof.layers):
                    Hk = self.hamiltonian.matter_constant(E_flat,
                                                          rho_gcm3=layer.rho_gcm3,
                                                          Ye=layer.Ye,
                                                          antineutrino=antineutrino)  # (B,N,N)
                    HLk = Hk * (dL_list[k] * KM)[:, xp.newaxis, xp.newaxis]  # (B,N,N)
                    Sk = linalg.matrix_exp((-1j) * HLk)  # (B,N,N)
                    S = Sk @ S  # pre-multiply: S_tot = S_k * S_tot
        else:
            H = self.hamiltonian.vacuum(E_flat, antineutrino=antineutrino) # (S*ns,N,N)
            HL = H * (L_flat * KM)[:, xp.newaxis, xp.newaxis]
            S = linalg.matrix_exp((-1j) * HL)

        if not use_sampling:
            # using true
            P = (xp.abs(S) ** 2).reshape(*center_shape, S.shape[-2], S.shape[-1])  # S+(N,N)
        else:
            # smearings
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

