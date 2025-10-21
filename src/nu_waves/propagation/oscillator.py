import numpy as np
from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.backends import make_numpy_backend
from nu_waves.matter.profile import MatterProfile
from nu_waves.utils.units import KM_TO_EVINV


class Oscillator:
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

        self.mixing_matrix = mixing_matrix
        self.m2_list = m2_list
        self.hamiltonian = None
        self.set_parameters(mixing_matrix=mixing_matrix, m2_list=m2_list)

        # samplers: callable(center_array, n_samples)
        self.energy_sampler = energy_sampler
        self.baseline_sampler = baseline_sampler
        self.n_samples = n_samples

    def set_parameters(self, mixing_matrix: np.ndarray =None, m2_list: np.ndarray =None):
        if mixing_matrix is None and m2_list is None:
            raise ValueError("Must specify either mixing_matrix or m2_list")

        if mixing_matrix is not None:
            self.mixing_matrix = mixing_matrix
        if m2_list is not None:
            self.m2_list = m2_list

        # rebuild hamiltonian
        self.hamiltonian = Hamiltonian(
            self.mixing_matrix, self.m2_list,
            backend=self.backend
        )


    def set_constant_density(self, rho_gcm3: float, Ye: float = 0.5):
        self._use_matter = True
        self._matter_args = (float(rho_gcm3), float(Ye))
        self._matter_profile = None

    def set_layered_profile(self, profile: MatterProfile):
        self._use_matter = True
        self._matter_profile = profile
        self._matter_args = None

    def use_vacuum(self):
        self._use_matter = False
        self._matter_args = None
        self._matter_profile = None

    # ----------------------------------------------------------------------
    def probability(self,
                    flavor_emit = None,
                    flavor_det = None,
                    L_km = 0.0,
                    E_GeV = 1.0,
                    antineutrino: bool = False
                    ):
        xp = self.backend.xp
        linalg = self.backend.linalg
        is_torch = self.backend.name == "torch"

        # ---------- normalize inputs & detect grid/pairs ----------
        L_in = xp.asarray(L_km, dtype=self.backend.dtype_real)
        E_in = xp.asarray(E_GeV, dtype=self.backend.dtype_real)

        # grid_mode = (L_in.ndim == 1 and E_in.ndim == 1 and L_in.size > 1 and E_in.size > 1)
        grid_mode = (
              L_in.ndim == 1 and E_in.ndim == 1
              and int(L_in.shape[0]) > 1 and int(E_in.shape[0]) > 1
        )

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
                if is_torch:
                    # slower
                    S = linalg.matrix_exp((-1j) * HL)
                else:
                    # HL is Hermitian (H Hermitian, scalar L). Use batched eigh:
                    evals, evecs = np.linalg.eigh(HL)  # (B,N), (B,N,N)
                    phases = np.exp((-1j) * evals).astype(self.backend.dtype_complex, copy=False)  # (B,N)
                    # S = evecs @ diag(phases) @ evecs^\dagger, fully vectorized:
                    S = evecs * phases[:, np.newaxis, :]  # (B,N,N)
                    S = S @ np.conjugate(evecs).transpose(0, 2, 1)  # (B,N,N)

            else:
                # layered profile
                prof = self._matter_profile
                # per-center ΔL_k arrays, each shaped like L_flat (broadcast-safe across grid or pairs)
                dL_list = prof.resolve_dL(self.backend.from_device(L_flat))  # resolve in host float
                dL_list = [xp.asarray(dL, dtype=self.backend.dtype_real) for dL in dL_list]

                # accumulate S_tot = S_K @ ... @ S_1 (source→detector order = list order)
                N = self.hamiltonian.U.shape[0]
                S = xp.eye(N, dtype=self.backend.dtype_complex)[xp.newaxis, ...]  # (1,N,N) seed

                if hasattr(S, "clone"):  # Torch tensors
                    S = xp.broadcast_to(S, (E_flat.shape[0], N, N)).clone()
                else:  # NumPy arrays
                    S = xp.broadcast_to(S, (E_flat.shape[0], N, N)).copy()

                # S = xp.broadcast_to(S, (E_flat.shape[0], N, N)).copy()  # (B,N,N) identities

                for k, layer in enumerate(prof.layers):

                    Hk = self.hamiltonian.matter_constant(
                        E_flat,
                        rho_gcm3=layer.rho_gcm3,
                        Ye=layer.Ye,
                        antineutrino=antineutrino
                    )  # (B,N,N)
                    HLk = Hk * (dL_list[k] * KM)[:, xp.newaxis, xp.newaxis]

                    if is_torch:
                        # slower
                        Sk = linalg.matrix_exp((-1j) * HLk)
                    else:
                        evals, evecs = np.linalg.eigh(HLk)
                        phases = np.exp((-1j) * evals).astype(self.backend.dtype_complex, copy=False)  # (B,N)
                        Sk = evecs * phases[:, np.newaxis, :]
                        Sk = Sk @ np.conjugate(evecs).transpose(0, 2, 1)  # (B,N,N)

                    S = Sk @ S  # pre-multiply: S_tot = S_k * S_tot
        else:
            if is_torch:
                # generic slow path (Torch or SciPy backend)
                H = self.hamiltonian.vacuum(E_flat, antineutrino=antineutrino)
                HL = H * (L_flat * KM)[:, xp.newaxis, xp.newaxis]
                S = linalg.matrix_exp((-1j) * HL)
            else:
                # --- VACUUM FAST PATH (NumPy) ---
                # S = U diag(exp(-i * m2 * L/E * const)) U^\dagger
                U = np.asarray(self.hamiltonian.U, dtype=self.backend.dtype_complex)  # (N,N)
                # Expect mass-squared as a length-N array on the Hamiltonian; if not available, fall back to H+eigh
                m2 = np.asarray(self.hamiltonian.m2_diag, dtype=self.backend.dtype_real)  # (N,)
                phase = (KM * L_flat / E_flat)[:, None] * m2[None, :]  # (B,N)
                phases = np.exp((-1j) * phase).astype(self.backend.dtype_complex, copy=False)  # (B,N)

                Uc = np.conjugate(U).T  # (N,N)
                U_phase = U[np.newaxis, :, :] * phases[:, np.newaxis, :]  # (B,N,N)
                S = U_phase @ Uc  # (B,N,N)

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
        a = _as_idx(flavor_emit, N)
        b = _as_idx(flavor_det, N)

        if flavor_emit is None and flavor_det is None:
            return self.backend.from_device(P)

        a_scalar = xp.isscalar(a)
        b_scalar = xp.isscalar(b)

        if not is_torch:
            # NumPy path (unchanged)
            out = P[..., b, a] if (a_scalar and b_scalar) or (a_scalar and not b_scalar) or (not a_scalar and b_scalar) else P[..., self.backend.xp.ix_(b, a)]
        else:
            # Torch path uses index_select (advanced indexing parity)
            to_idx = lambda x: xp.asarray(x, int)
            if a_scalar and b_scalar:
                out = P[..., int(b), int(a)]
            elif a_scalar and not b_scalar:
                out = P.index_select(-2, to_idx(b))[..., int(a)]
            elif not a_scalar and b_scalar:
                out = P.index_select(-1, to_idx(a)).index_select(-2, to_idx(b))
            else:
                out = P.index_select(-2, to_idx(b)).index_select(-1, to_idx(a))

        return self.backend.from_device(out)

    def adiabatic_mass_fractions(
          self,
          E_GeV: float | np.ndarray,
          profile,  # SolarProfile-like: rho_gcm3(r), Ye(r)
          r_km: np.ndarray,  # (n,) path radii (production → surface), any monotonic order
          alpha: int = 0,
          antineutrino: bool = False,
    ):
        """
        Returns:
          F : (n, N) phase-averaged fractions in vacuum mass eigenstates,
              aligned with the input r_km order.
        """
        xp, linalg = self.backend.xp, self.backend.linalg
        N = self.hamiltonian.U.shape[0]
        U = self.hamiltonian.U

        r_in = np.asarray(r_km, float)
        if r_in.ndim != 1 or r_in.size < 2:
            raise ValueError("r_km must be a 1D array with length ≥ 2.")

        # Work internally with increasing radius; remember if we reversed
        rev = r_in[0] > r_in[-1]
        r_path = r_in[::-1] if rev else r_in

        rho_np = profile.rho_gcm3(r_path)
        Ye_np = profile.Ye(r_path)
        H_list = []
        for k in range(r_path.size):
            Hk = self.hamiltonian.matter_constant(
                E_GeV, rho_gcm3=float(rho_np[k]), Ye=float(Ye_np[k]),
                antineutrino=antineutrino
            )
            H_list.append(Hk[0] if Hk.ndim == 3 else Hk)  # ensure (N,N)

        H = xp.stack(H_list, axis=0)  # (n,N,N)
        evals, evecs = linalg.eigh(H)  # (n,N), (n,N,N)

        # Mode tracking (reorder columns for continuity + phase alignment)
        V_tracked = [evecs[0]]
        for k in range(1, evecs.shape[0]):
            V_prev, V_cur = V_tracked[-1], evecs[k]
            O = xp.abs(xp.swapaxes(xp.conj(V_prev), -1, -2) @ V_cur)  # (N,N)
            order, used = [], set()
            for i in range(N):
                j = int(xp.argmax(O[i, :]).item())
                while j in used:
                    O[i, j] = -1.0
                    j = int(xp.argmax(O[i, :]).item())
                order.append(j);
                used.add(j)
            Vc = V_cur[:, order]
            for j in range(N):
                phase = xp.angle(xp.einsum("i,i->", xp.conj(V_prev[:, j]), Vc[:, j]))
                Vc[:, j] *= xp.exp(-1j * phase)
            V_tracked.append(Vc)
        V_tracked = xp.stack(V_tracked, axis=0)  # (n,N,N)

        # Production flavor decomposition in matter basis (weights conserved adiabatically)
        e_alpha = xp.zeros((N,), dtype=self.backend.dtype_complex);
        e_alpha[alpha] = 1.0
        c0 = xp.swapaxes(xp.conj(V_tracked[0]), -1, -2) @ e_alpha
        w = xp.abs(c0) ** 2  # (N,)

        # Overlap with vacuum mass eigenvectors; phase-averaged fractions
        M = xp.abs(xp.swapaxes(xp.conj(U), -1, -2) @ V_tracked) ** 2  # (n,N,N)
        F = M @ w  # (n,N)

        if rev:
            F = F[::-1, :]  # restore original r_km ordering

        return self.backend.from_device(F)

    def initial_mass_composition(
          self,
          alpha: int,
          basis: str = "vacuum",  # "vacuum" (no matter), "matter", or "vacuum_from_matter"
          E_GeV: float | None = None,
          profile=None,
          r_emit_km: float | None = None,
          antineutrino: bool = False,
    ):
        """
        Returns a length-N vector of fractions.

        basis="vacuum":
            Return |U_{alpha i}|^2 (no matter input needed).
        basis="matter":
            Return w_matter[j] = |<nu_j^m(r_emit)|nu_alpha>|^2, needs E, profile, r_emit_km.
        basis="vacuum_from_matter":
            Adiabatic-consistent 'initial' vacuum-mass fractions at emission:
            F0[i] = sum_j |<nu_i^vac|nu_j^m(r_emit)>|^2 * w_matter[j].
        """
        xp = self.backend.xp
        U = self.hamiltonian.U  # (N,N)
        N = U.shape[0]

        if basis == "vacuum":
            e_alpha = xp.zeros((N,), dtype=self.backend.dtype_complex);
            e_alpha[alpha] = 1.0
            F0 = xp.abs(xp.swapaxes(xp.conj(U), -1, -2) @ e_alpha) ** 2  # |U^† e_alpha|^2
            return self.backend.from_device(F0)

        # the two matter-aware options need E, profile, r_emit_km
        if E_GeV is None or profile is None or r_emit_km is None:
            raise ValueError("E_GeV, profile, and r_emit_km are required for matter-aware bases.")

        rho = float(np.asarray(profile.rho_gcm3([r_emit_km]), float)[0])
        Ye = float(np.asarray(profile.Ye([r_emit_km]), float)[0])
        H = self.hamiltonian.matter_constant(E_GeV, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)
        if H.ndim == 3: H = H[0]  # (N,N)

        evals, V = self.backend.linalg.eigh(H)  # columns = |nu_j^m>
        e_alpha = xp.zeros((N,), dtype=self.backend.dtype_complex);
        e_alpha[alpha] = 1.0
        c = xp.swapaxes(xp.conj(V), -1, -2) @ e_alpha
        w_matter = xp.abs(c) ** 2  # (N,)

        if basis == "matter":
            return self.backend.from_device(w_matter)

        if basis == "vacuum_from_matter":
            M = xp.abs(xp.swapaxes(xp.conj(U), -1, -2) @ V) ** 2  # (N,N)
            F0 = M @ w_matter
            return self.backend.from_device(F0)

        raise ValueError("basis must be 'vacuum', 'matter', or 'vacuum_from_matter'")

    def adiabatic_mass_fractions_from_emission(
          self,
          E_GeV: float | np.ndarray,
          profile,  # SolarProfile-like: rho_gcm3(r), Ye(r), R_sun_km
          r_emit_km: float,
          s_km: np.ndarray,  # (n,) propagation lengths from emission, km (>=0), monotonic
          alpha: int = 0,
          antineutrino: bool = False,
    ):
        """
        Phase-averaged vacuum mass fractions along a path defined by distances s_km from r_emit_km.
        Returns F with shape (n, N), aligned with s_km (i.e., with r = r_emit_km + s_km).
        """
        s = np.asarray(s_km, float)
        if s.ndim != 1 or s.size < 2:
            raise ValueError("s_km must be a 1D array with length ≥ 2.")
        if np.any(s < 0):
            raise ValueError("s_km must be non-negative.")
        # monotonic check (allow equal last samples)
        if np.any(np.diff(s) < 0):
            raise ValueError("s_km must be non-decreasing.")

        R_sun = getattr(profile, "R_sun_km", np.inf)
        r_path = r_emit_km + s
        # clamp to surface to avoid overshooting
        r_path = np.clip(r_path, 0.0, R_sun)

        return self.adiabatic_mass_fractions(
            E_GeV=E_GeV, profile=profile, r_km=r_path,
            alpha=alpha, antineutrino=antineutrino
        )
