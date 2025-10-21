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
                 backend=None,
                 use_exponentiation=False, # slower
                 ):
        self.backend = backend or make_numpy_backend()

        self.mixing_matrix = mixing_matrix
        self.m2_list = m2_list
        self.hamiltonian = None
        self.set_parameters(mixing_matrix=mixing_matrix, m2_list=m2_list)

        self._use_matter = False
        self.use_exponentiation = use_exponentiation

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

    def set_backend(self, backend):
        self.backend = backend or make_numpy_backend()
        self.hamiltonian.set_backend(self.backend)

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

    # helpers to make sure the pullback from the GPU memory happens once
    def propagate_state(self, flavor_emit, L_km, E_GeV, antineutrino=False):
        return self.backend.to_device(
            self._propagate_state(flavor_emit=flavor_emit, L_km=L_km, E_GeV=E_GeV, antineutrino=antineutrino)
        )

    def probability(self, L_km, E_GeV, flavor_emit=None, flavor_det=None, antineutrino=False):
        return self.backend.to_device(
            self._probability(
                L_km=L_km, E_GeV=E_GeV, flavor_emit=flavor_emit, flavor_det=flavor_det, antineutrino=antineutrino
            )
        )

    # private
    def _generate_initial_state(self, flavor_emit, E_GeV, antineutrino=False):
        """
        Generate the initial state(s) |psi(0)> corresponding to the chosen
        flavor(s) in the appropriate basis (flavour or matter).

        Parameters
        ----------
        flavor_emit : int or array-like
            Indices of emitting flavour(s). Example: 0=e, 1=mu, 2=tau.
        E_GeV : float or 1D array
            Neutrino energies. Used only if matter effects are enabled.
        antineutrino : bool
            Use the conjugate matter Hamiltonian if True.

        Returns
        -------
        psi0 : array
            Initial state(s), complex array with shape:
              - (N,)              for a single flavour
              - (nEmit, N)        for multiple flavours
              - (nE, N) or (nE, nEmit, N) if matter effects cause energy dependence
        """
        xp = self.backend.xp
        N = self.hamiltonian.U.shape[0]

        # ---------- normalize indices ----------
        def _as_idx(x, N):
            if x is None:
                return xp.arange(N)
            x = xp.asarray(x)
            return int(x) if x.ndim == 0 else x

        a_idx = _as_idx(flavor_emit, N)
        a_scalar = xp.isscalar(a_idx)
        if a_scalar:
            a_idx = [a_idx]

        # ---------- vacuum or matter basis ----------
        if not getattr(self, "_use_matter", False):
            # --- simple flavour basis ---
            psi0 = xp.zeros((len(a_idx), N), dtype=self.backend.dtype_complex)
            psi0[xp.arange(len(a_idx)), xp.asarray(a_idx, int)] = 1.0
            if len(a_idx) == 1:
                psi0 = psi0[0]  # (N,)
            return self.backend.from_device(psi0)

        # --- matter case: energy dependence ---
        E_in = xp.asarray(E_GeV, dtype=self.backend.dtype_real)
        if E_in.ndim == 0:
            E_in = E_in.reshape(1)

        if getattr(self, "_matter_profile", None) is None:
            rho, Ye = self._matter_args
            H = self.hamiltonian.matter_constant(E_in, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)
            evals, V_np = np.linalg.eigh(self.backend.from_device(H))
            V = xp.asarray(V_np, dtype=self.backend.dtype_complex)  # (nE,N,N)
        else:
            # use density of first (production) layer
            layer = self._matter_profile.layers[0]
            rho, Ye = layer.rho_gcm3, layer.Ye
            H = self.hamiltonian.matter_constant(E_in, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)
            evals, V_np = np.linalg.eigh(self.backend.from_device(H))
            V = xp.asarray(V_np, dtype=self.backend.dtype_complex)

        # ---------- project flavour states into matter basis ----------
        # V transforms mass→flavour, so its conjugate columns correspond to flavour→mass
        Vc = xp.conjugate(V)

        if V.ndim == 2:
            # single energy
            psi0 = Vc[:, a_idx]  # (N, nEmit)
            psi0 = xp.moveaxis(psi0, -1, 0)  # (nEmit, N)
        else:
            # multiple energies: (nE,N,N)
            psi0 = Vc[:, :, a_idx]  # (nE,N,nEmit)
            psi0 = xp.moveaxis(psi0, -1, -2)  # (nE,nEmit,N)

        # --- squeeze scalar axes for consistency ---
        if psi0.shape[0] == 1 and psi0.ndim == 3:
            psi0 = psi0[0]
        if psi0.ndim == 2 and psi0.shape[0] == 1:
            psi0 = psi0[0]

        return self.backend.from_device(psi0)

    def _propagate_state(self, psi, L_km, E_GeV, antineutrino=False):
        """
        Return psi(L) = S(L,E) @ psi(0)

        Parameters
        ----------
        psi : array-like
            Initial state(s) in flavour basis. Can be:
              - shape (N,)             for a single state
              - shape (..., N)         for multiple states (e.g. (nEmit, N))
        L_km, E_GeV : float or 1D array
            Baseline(s) and energy(ies). Pairwise semantics: if both are arrays,
            their lengths must match.
        antineutrino : bool
            Use conjugate Hamiltonian if True.

        Returns
        -------
        psi_out : array
            Propagated state(s) in flavour basis, with shape
              (..., N) or (..., nEmit, N), matching psi and inputs.
        """
        xp = self.backend.xp
        linalg = self.backend.linalg

        # ---------- normalize inputs ----------
        L_in = xp.asarray(L_km, dtype=self.backend.dtype_real)
        E_in = xp.asarray(E_GeV, dtype=self.backend.dtype_real)

        if L_in.ndim == 0:
            L_in = L_in.reshape(1)
        if E_in.ndim == 0:
            E_in = E_in.reshape(1)

        # ---------- enforce pairwise semantics ----------
        if L_in.size == 1 and E_in.size > 1:
            Lc = xp.broadcast_to(L_in, E_in.shape)
            Ec = E_in
        elif E_in.size == 1 and L_in.size > 1:
            Lc = L_in
            Ec = xp.broadcast_to(E_in, L_in.shape)
        else:
            if L_in.size != E_in.size:
                raise ValueError(
                    f"Length mismatch: L_km has {L_in.size}, E_GeV has {E_in.size}. "
                    "They must match for pairwise propagation."
                )
            Lc, Ec = L_in, E_in

        center_shape = Lc.shape
        use_sampling = (self.energy_sampler is not None) or (self.baseline_sampler is not None)

        # ---------- prepare flattened arrays ----------
        if not use_sampling:
            E_flat = Ec.reshape(-1)
            L_flat = Lc.reshape(-1)
        else:
            ns = int(max(1, self.n_samples))

            def _tile(x):
                return xp.repeat_last(x, ns)

            Es = self.energy_sampler(Ec, ns) if self.energy_sampler else _tile(Ec)
            Ls = self.baseline_sampler(Lc, ns) if self.baseline_sampler else _tile(Lc)
            E_flat = Es.reshape(-1)
            L_flat = Ls.reshape(-1)

        KM = xp.asarray(KM_TO_EVINV, dtype=self.backend.dtype_real)

        # ---------- evolution operator ----------
        if getattr(self, "_use_matter", False):
            if getattr(self, "_matter_profile", None) is None:
                rho, Ye = self._matter_args
                H = self.hamiltonian.matter_constant(E_flat, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)
                HL = H * (L_flat * KM)[:, xp.newaxis, xp.newaxis]
                if self.use_exponentiation:
                    S = linalg.matrix_exp((-1j) * HL)
                else:
                    evals, evecs = xp.linalg.eigh(HL)
                    phases = xp.exp((-1j) * evals).astype(self.backend.dtype_complex, copy=False)
                    S = evecs * phases[:, xp.newaxis, :]
                    S = S @ xp.conjugate(evecs).transpose(0, 2, 1)
            else:
                prof = self._matter_profile
                dL_list = prof.resolve_dL(self.backend.from_device(L_flat))
                dL_list = [xp.asarray(dL, dtype=self.backend.dtype_real) for dL in dL_list]

                N = self.hamiltonian.U.shape[0]
                S = xp.eye(N, dtype=self.backend.dtype_complex)[xp.newaxis, ...]
                S = xp.broadcast_to(S, (E_flat.shape[0], N, N)).copy()

                for k, layer in enumerate(prof.layers):
                    Hk = self.hamiltonian.matter_constant(
                        E_flat, rho_gcm3=layer.rho_gcm3, Ye=layer.Ye, antineutrino=antineutrino
                    )
                    HLk = Hk * (dL_list[k] * KM)[:, xp.newaxis, xp.newaxis]
                    if self.use_exponentiation:
                        Sk = linalg.matrix_exp((-1j) * HLk)
                    else:
                        evals, evecs = xp.linalg.eigh(HLk)
                        phases = xp.exp((-1j) * evals).astype(self.backend.dtype_complex, copy=False)
                        Sk = evecs * phases[:, xp.newaxis, :]
                        Sk = Sk @ xp.conjugate(evecs).transpose(0, 2, 1)
                    S = Sk @ S
        else:
            # vacuum
            if self.use_exponentiation:
                H = self.hamiltonian.vacuum(E_flat, antineutrino=antineutrino)
                HL = H * (L_flat * KM)[:, xp.newaxis, xp.newaxis]
                S = linalg.matrix_exp((-1j) * HL)
            else:
                U = xp.asarray(self.hamiltonian.U, dtype=self.backend.dtype_complex)
                m2 = xp.asarray(self.hamiltonian.m2_diag, dtype=self.backend.dtype_real)
                phase = (KM * L_flat / E_flat)[:, None] * m2[None, :]
                phases = xp.exp((-1j) * phase).astype(self.backend.dtype_complex, copy=False)
                Uc = xp.conjugate(U).T
                U_phase = U[xp.newaxis, :, :] * phases[:, xp.newaxis, :]
                S = U_phase @ Uc

        # ---------- apply initial state(s) ----------
        psi = xp.asarray(psi, dtype=self.backend.dtype_complex)
        if psi.ndim == 1:
            # single state (N,)
            psi_out = xp.einsum("bij,j->bi", S, psi)  # (B,N)
        else:
            # multiple states (...,N)
            psi_out = xp.einsum("bij,kj->bki", S, psi)  # (B,nPsi,N)

        # ---------- reshape back ----------
        if not use_sampling:
            psi_out = psi_out.reshape(*center_shape, *psi_out.shape[1:])
        else:
            psi_out = psi_out.reshape(*center_shape, ns, *psi_out.shape[1:]).mean(axis=-3)

        # ---------- squeeze scalar axes ----------
        if L_in.size == 1 and E_in.size == 1:
            psi_out = psi_out[0]
        elif psi_out.shape[0] == 1:
            psi_out = psi_out[0]

        return psi_out

    def _project_state(self, psi, E_GeV=None, antineutrino=None):
        """
        Project a propagated state psi(..., N) from flavour basis onto
        the mass/matter eigenbasis.

        psi may have arbitrary leading dimensions (e.g. (nE, nEmit, N)).
        Returns:
            a  : same leading shape as psi, last dim = N (mass amplitudes)
            V  : eigenvector matrix (N,N) or batch (nE,N,N)
        """
        xp = self.backend.xp

        # choose projection matrix V (vacuum or matter)
        if not self._use_matter:
            V = xp.asarray(self.hamiltonian.U, dtype=self.backend.dtype_complex)  # (N,N)
        else:
            assert E_GeV is not None, "Need E_GeV to diagonalize Hamiltonian in matter."
            rho, Ye = (
                self._matter_args if getattr(self, "_matter_profile", None) is None
                else (self._matter_profile.layers[-1].rho_gcm3,
                      self._matter_profile.layers[-1].Ye)
            )
            import numpy as np
            E_arr = xp.asarray(E_GeV, dtype=self.backend.dtype_real).reshape(-1)
            H = self.hamiltonian.matter_constant(E_arr, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)
            evals, V_np = np.linalg.eigh(self.backend.from_device(H))
            V = xp.asarray(V_np, dtype=self.backend.dtype_complex)  # (nE,N,N)

        # Determine number of flavour components
        N = psi.shape[-1]
        lead_shape = psi.shape[:-1]  # e.g. (10,3)
        psi_view = psi.reshape(-1, N)  # flatten leading dims only for batched multiplication

        # --- Apply projection ---
        if V.ndim == 2:
            # one global matrix (vacuum)
            a_flat = (xp.conjugate(V).T @ psi_view.T).T  # (B,N)
        else:
            # batch of matrices (one per energy)
            # we need to broadcast correctly: (nE,N,N) with psi_view (nE*nEmit,N)
            nE = V.shape[0]
            nEmit = psi.size // (nE * N)
            psi_view = psi.reshape(nE, nEmit, N)
            a_flat = xp.einsum("bij,bkj->bki", xp.conjugate(V), psi_view)  # (nE,nEmit,N)

        # --- reshape back to original leading shape ---
        a = a_flat.reshape(*lead_shape, N)

        return a, V

    def _probability(self, L_km, E_GeV, flavor_emit=None, flavor_det=None, antineutrino=False):
        return self._probability_split_adiabatic(
            L_km=L_km, E_GeV=E_GeV, flavor_emit=flavor_emit, flavor_det=flavor_det, antineutrino=antineutrino
        )["total"]

    def _probability_split_adiabatic(self, L_km, E_GeV,
                                     flavor_emit=None, flavor_det=None,
                                     antineutrino=False):
        xp = self.backend.xp

        E_GeV = xp.broadcast_to(E_GeV, xp.shape(L_km)) if xp.ndim(L_km) == 1 and xp.ndim(E_GeV) == 0 else E_GeV

        # --- Step 1: propagate and project onto mass basis ---
        psi0 = self._generate_initial_state(flavor_emit=flavor_emit, E_GeV=E_GeV, antineutrino=antineutrino)
        psi = self._propagate_state(psi=psi0, L_km=L_km, E_GeV=E_GeV, antineutrino=antineutrino)
        a, V = self._project_state(
            psi=psi,
            E_GeV=E_GeV if self._use_matter else None,
            antineutrino=antineutrino if self._use_matter else None
        )
        # a: (E,N) or (E,A,N)
        N = a.shape[-1]
        lead_shape = a.shape[:-1]
        nE = lead_shape[0]
        nEmit = 1 if a.ndim == 2 else lead_shape[1]

        # --- Step 2: indices (keep scalar flags BEFORE list conversion) ---
        def _as_idx(x, N):
            if x is None:
                return xp.arange(N)
            x = xp.asarray(x)
            return int(x) if x.ndim == 0 else x

        a_idx_raw = _as_idx(flavor_emit, N)
        b_idx_raw = _as_idx(flavor_det, N)
        a_scalar_in = xp.isscalar(a_idx_raw)
        b_scalar_in = xp.isscalar(b_idx_raw)
        a_idx = [a_idx_raw] if a_scalar_in else a_idx_raw
        b_idx = [b_idx_raw] if b_scalar_in else b_idx_raw
        nDet = len(b_idx)

        # --- Step 3: build flavour projectors ---
        if V.ndim == 2:
            Vb = V[b_idx, :]  # (B,N)
            Vb = xp.broadcast_to(Vb, (nE, nDet, N))  # (E,B,N)
        else:
            Vb = V[:, b_idx, :]  # (E,B,N)

        # --- Step 4: amplitudes & probabilities ---
        if a.ndim == 2:
            # single emitter: a (E,N) -> A_i (E,B,N)
            A_i = Vb * a[:, None, :]  # (E,B,N)
            P_incoh = xp.sum(xp.abs(A_i) ** 2, axis=-1)  # (E,B)
            P_total = xp.abs(xp.sum(A_i, axis=-1)) ** 2  # (E,B)
            P_int = P_total - P_incoh  # (E,B)
        else:
            # multiple emitters: a (E,A,N) -> A_i (E,B,A,N)
            A_i = Vb[:, :, None, :] * a[:, None, :, :]  # (E,B,A,N)
            P_incoh = xp.sum(xp.abs(A_i) ** 2, axis=-1)  # (E,B,A)
            P_total = xp.abs(xp.sum(A_i, axis=-1)) ** 2  # (E,B,A)
            P_int = P_total - P_incoh  # (E,B,A)

        # --- Step 5: if detector is scalar, squeeze the B axis (axis=1) ---
        if b_scalar_in:
            # (E,B)   -> (E,)     ; (E,B,A) -> (E,A)
            P_total = xp.squeeze(P_total, axis=1)
            P_incoh = xp.squeeze(P_incoh, axis=1)
            P_int = xp.squeeze(P_int, axis=1)

        return {
            "total": self.backend.from_device(P_total),
            "incoherent": self.backend.from_device(P_incoh),
            "interference": self.backend.from_device(P_int),
        }

    # def _get_mass_fractions_after(self, flavor_emit, L_km, E_GeV, antineutrino=False, return_amplitudes=False, psi=None):
    #
    #     if psi is None:
    #         psi = self._propagate_state(flavor_emit, L_km, E_GeV, antineutrino)
    #
    #     xp = self.backend.xp
    #
    #     return (xp.abs(a) ** 2).astype(self.backend.dtype_real, copy=False)

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
