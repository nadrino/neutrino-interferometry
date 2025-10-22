import numpy as np
from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.backends import make_numpy_backend
from nu_waves.matter.profile import MatterProfile
from nu_waves.utils.units import KM_TO_EVINV
from nu_waves.utils.units import GEV_TO_EV


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
    def generate_initial_state(self, flavor_emit, E_GeV, antineutrino=False):
        return self.backend.from_device(
            self._generate_initial_state(flavor_emit=flavor_emit, E_GeV=E_GeV, antineutrino=antineutrino)
        )

    def propagate_state(self, psi, L_km, E_GeV, antineutrino=False):
        return self.backend.from_device(
            self._propagate_state(psi=psi, L_km=L_km, E_GeV=E_GeV, antineutrino=antineutrino)
        )

    def probability(self, L_km, E_GeV, flavor_emit=None, flavor_det=None, antineutrino=False):
        L, E = self._generate_L_and_E_arrays(L_km, E_GeV)
        flavor_emit = self._format_flavor_arg(flavor_emit)
        flavor_det = self._format_flavor_arg(flavor_det)
        out = self._probability(
            L=L, E=E,
            flavor_emit=flavor_emit,
            flavor_det=flavor_det,
            antineutrino=antineutrino
        )
        return self.backend.from_device(self._squeeze_array(out))

    def probability_sampled(self, L_true_km, E_true_GeV, flavor_emit=None, flavor_det=None, antineutrino=False,
                            L_sample_fct=None, E_sample_fct=None, nSamples=100):
        L, E = self._format_arrays(L_true_km, E_true_GeV)

        # out = self._probability_sampled(L_true_km=L, E_true_GeV=E,
        #                               flavor_emit=flavor_emit, flavor_det=flavor_det,
        #                               antineutrino=antineutrino,
        #                               L_sample_fct=L_sample_fct, E_sample_fct=E_sample_fct, nSamples=nSamples)
        #
        # return self.backend.from_device(out)

    def _squeeze_array(self, x, preserve_axes=None):
        """
        Remove all dims of size 1.
        If preserve_axes is provided (int or iterable, supports negative indices),
        those axes are kept even if they have size 1.
        """
        # fast path when we don't need to preserve anything
        if preserve_axes is None or (isinstance(preserve_axes, (list, tuple)) and len(preserve_axes) == 0):
            # NumPy and Torch both implement .squeeze()
            return x.squeeze()

        # normalize preserve set with positive indices
        ndim = x.ndim if hasattr(x, "ndim") else x.dim()
        if isinstance(preserve_axes, int):
            preserve_axes = (preserve_axes,)
        pres = set()
        for a in preserve_axes:
            a = int(a)
            if a < 0:
                a += ndim
            if a < 0 or a >= ndim:
                raise IndexError(f"preserve axis {a} out of range for ndim={ndim}")
            pres.add(a)

        # build new shape (keep any non-1 dims or preserved axes)
        shape = [int(s) for s in x.shape]
        new_shape = [s for i, s in enumerate(shape) if (s != 1) or (i in pres)]

        # if everything would be squeezed, return a scalar (0-d)
        if len(new_shape) == 0:
            return x.reshape(())

        # reshape is metadata-only in NumPy, and in Torch returns a view if possible
        return x.reshape(new_shape)

    def _format_flavor_arg(self, arg):
        """
        Normalize to list[int].
        Allowed: None → full range [0..n_flavors-1], int, list[int].
        Disallowed: anything else.
        """
        xp = self.backend.xp
        if arg is None:
            return list(range(int(self.hamiltonian.n_flavors)))

        if isinstance(arg, int):
            out = [int(arg)]
        elif isinstance(arg, list) and all(isinstance(x, int) for x in arg):
            out = [int(x) for x in arg]
        else:
            raise TypeError(f"Flavor arg must be None, int, or list of int.")

        return out

    def _format_arrays(self, L_km, E_GeV):
        """
        Accept scalars or 1D arrays (Torch or NumPy).
        Returns L_vec, E_vec shaped (N,), with broadcasting if one is scalar.
        """
        xp = self.backend.xp
        L = xp.as_real(L_km)
        E = xp.as_real(E_GeV)

        # Convert all in eV
        L = L * KM_TO_EVINV
        E = E * GEV_TO_EV

        nL = xp.ndim(L)
        nE = xp.ndim(E)

        # Promote scalars to vectors where needed
        if nL == 0 and nE == 0:
            L = L.reshape(1)
            E = E.reshape(1)
        elif nL == 0 and nE == 1:
            L = xp.ones_like(E) * L
        elif nL == 1 and nE == 0:
            E = xp.ones_like(L) * E
        elif nL == 1 and nE == 1:
            if L.shape != E.shape:
                raise ValueError(f"L and E must have the same length; got {L.shape} vs {E.shape}.")
        else:
            raise ValueError("L_km and E_GeV must be scalar or 1D.")

        # Ensure contiguous 1D vectors
        L = L.reshape(-1)
        E = E.reshape(-1)
        return L, E

    def _generate_initial_state(self, flavor_emit, E, antineutrino=False):
        """
        PRIVATE core.
        Inputs (assumed valid; no checks performed):
          - E: xp 1D array with shape (nE,), energies in eV.
          - flavor_emit: xp 1D array with shape (nF,), flavour-basis amplitudes
                         (e.g. one-hot for a pure flavour).
        Output:
          - psi0: xp complex array with shape (nE, nF).
            In vacuum mode:      flavour-basis, tiled over energies.
            In matter mode:      matter-eigenbasis projection V(E)^\\dagger * flavor_emit.
        """
        xp = self.backend.xp
        dtype_c = self.backend.dtype_complex
        N = self.hamiltonian.n_flavors  # nF
        nE = E.shape[0]

        # Coerce list -> backend array with complex dtype
        a_f = xp.asarray(flavor_emit, dtype=dtype_c)  # (nF,)

        if not self._use_matter:
            # Vacuum: psi0 is flavour-basis and identical for all energies
            # (nE,1) * (1,nF) -> (nE,nF)
            psi0 = xp.ones((nE, 1), dtype=dtype_c) * a_f[xp.newaxis, :]
            return psi0

        # Matter: use first-layer (production) density or constant-density args
        if getattr(self, "_matter_profile", None) is None:
            rho, Ye = self._matter_args
        else:
            layer = self._matter_profile.layers[0]
            rho, Ye = layer.rho_gcm3, layer.Ye

        # H(E) diagonalization on CPU, then back to backend
        H = self.hamiltonian.matter_constant(E, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)  # (nE,N,N)
        import numpy as np
        _, V_np = np.linalg.eigh(self.backend.from_device(H))  # V_np: (nE,N,N), mass->flavour
        V = xp.asarray(V_np, dtype=dtype_c)

        # psi0(E) = V(E)^\dagger * a_f  -> (nE,N,N) @ (N,1) -> (nE,N,1) -> (nE,N)
        a_col = a_f.reshape(N, 1)
        psi0 = (xp.conjugate(V) @ a_col)[..., 0]
        return psi0

    def _propagate_state(self, psi, L, E, antineutrino=False):
        """
        PRIVATE core.
        Assumptions (no checks):
          - psi: xp complex array, shape (nE, nF)  [flavour basis at x=0]
          - L:   xp real array,   shape (nE,)      [in eV^-1]
          - E:   xp real array,   shape (nE,)      [in eV]
        Returns:
          - psi_out: xp complex array, shape (nE, nF)  [flavour basis at x=L]
        """
        xp = self.backend.xp
        linalg = self.backend.linalg
        dtype_c = self.backend.dtype_complex
        dtype_r = self.backend.dtype_real

        # ---------- evolution operator S (nE, nF, nF) ----------
        if self._use_matter:
            if self._matter_profile is None:
                # constant-density MSW
                rho, Ye = self._matter_args
                H = self.hamiltonian.matter_constant(E, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)  # (nE,nF,nF)
                HL = H * L[:, xp.newaxis, xp.newaxis]  # (nE,nF,nF)
                if self.use_exponentiation:
                    S = linalg.matrix_exp((-1j) * HL)  # (nE,nF,nF)
                else:
                    evals, evecs = xp.linalg.eigh(HL)  # HL = V diag(e) V^†
                    phases = xp.exp((-1j) * evals)  # (nE,nF)
                    S = (evecs * phases[:, xp.newaxis, :]) @ xp.conjugate(evecs).transpose(0, 2, 1)  # (nE,nF,nF)
            else:
                # layered matter profile; dL_list[k] is (nE,) and in the SAME UNITS as L (eV^-1)
                prof = self._matter_profile
                dL_list = prof.resolve_dL(L)  # assumed (list of xp arrays) with shape (nE,)
                nF = self.hamiltonian.U.shape[0]
                S = xp.eye(nF, dtype=dtype_c)[xp.newaxis, ...]
                S = xp.broadcast_to(S, (E.shape[0], nF, nF)).copy()
                for k, layer in enumerate(prof.layers):
                    Hk = self.hamiltonian.matter_constant(E, rho_gcm3=layer.rho_gcm3, Ye=layer.Ye,
                                                          antineutrino=antineutrino)  # (nE,nF,nF)
                    HLk = Hk * dL_list[k][:, xp.newaxis, xp.newaxis]  # (nE,nF,nF)
                    if self.use_exponentiation:
                        Sk = linalg.matrix_exp((-1j) * HLk)
                    else:
                        evals, evecs = xp.linalg.eigh(HLk)
                        phases = xp.exp((-1j) * evals)
                        Sk = (evecs * phases[:, xp.newaxis, :]) @ xp.conjugate(evecs).transpose(0, 2, 1)
                    S = Sk @ S
        else:
            # vacuum
            if self.use_exponentiation:
                H = self.hamiltonian.vacuum(E, antineutrino=antineutrino)  # (nE,nF,nF)
                HL = H * L[:, xp.newaxis, xp.newaxis]
                S = linalg.matrix_exp((-1j) * HL)
            else:
                U = xp.asarray(self.hamiltonian.U, dtype=dtype_c)  # (nF,nF)
                m2 = xp.asarray(self.hamiltonian.m2_diag, dtype=dtype_r)  # (nF,)  [eV^2]
                phase = 0.5 * (L / E)[:, None] * m2[None, :]  # (nE,nF)
                phases = xp.exp((-1j) * phase)  # (nE,nF)
                Uc = xp.conjugate(U).T
                U_phase = U[xp.newaxis, :, :] * phases[:, xp.newaxis, :]  # (nE,nF,nF)
                S = U_phase @ Uc  # (nE,nF,nF)

        # ---------- apply initial state(s): psi_out(b,i) = sum_j S(b,i,j) psi(b,j) ----------
        psi = xp.asarray(psi, dtype=dtype_c)  # (nE,nF)
        psi_out = xp.einsum("bij,bj->bi", S, psi)  # (nE,nF)
        return psi_out

    def _project_state(self, psi, E=None, antineutrino=False):
        """
        PRIVATE core.
        Assumptions:
          - psi: xp complex array, shape (nE, nF) in flavour basis
          - E:   xp real array,   shape (nE,) in eV (used only if self._use_matter)
        Returns:
          - a: xp complex array, shape (nE, nF)  [mass/matter amplitudes]
          - V: eigenvector matrix U_m (vacuum: (nF,nF); matter: (nE,nF,nF))
               (V maps mass->flavour; projection uses V^†)
        """
        xp = self.backend.xp
        dtype_c = self.backend.dtype_complex

        if not self._use_matter:
            # Vacuum: single mixing matrix
            V = xp.asarray(self.hamiltonian.U, dtype=dtype_c)  # (nF, nF)
            a = psi @ xp.conjugate(V)  # (nE, nF), since V^† right-multiplies rows
            return a, V

        # Matter: use last-layer density (detection layer) or constant-density args
        if getattr(self, "_matter_profile", None) is None:
            rho, Ye = self._matter_args
        else:
            layer = self._matter_profile.layers[-1]
            rho, Ye = layer.rho_gcm3, layer.Ye

        H = self.hamiltonian.matter_constant(E, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)  # (nE,nF,nF)

        import numpy as np
        _, V_np = np.linalg.eigh(self.backend.from_device(H))  # CPU eigvecs, (nE,nF,nF)
        V = xp.asarray(V_np, dtype=dtype_c)

        # Row-wise projection: a_b = psi_b @ V_b^†
        a = xp.einsum("bi,bij->bj", psi, xp.conjugate(V))  # (nE, nF)
        return a, V

    def _probability_split_adiabatic(self, L, E, flavor_emit=None, flavor_det=None, antineutrino=False):
        xp = self.backend.xp
        nF = self.hamiltonian.n_flavors
        nE = E.shape[0]
        nFe = len(flavor_emit)
        nFd = len(flavor_det)

        # --- Step 1: propagate and project onto mass basis ---
        psi0 = self._generate_initial_state(flavor_emit=flavor_emit, E=E, antineutrino=antineutrino)
        print(f"psi0 = {psi0}")
        psi = self._propagate_state(psi=psi0, L=L, E=E, antineutrino=antineutrino)
        print(f"psi = {psi}")
        a, V = self._project_state(
            psi=psi,
            E=E if self._use_matter else None,
            antineutrino=antineutrino if self._use_matter else None
        )
        print(f"V.shape = {V.shape}")
        print(f"V = {V}")
        print(f"a.shape = {a.shape}")
        print(f"a = {a}")

        # a: (nE, nFe)
        # V: (nFe, nFe) [vacuum] or (nE, nFe, nFe) [matter]
        xp = self.backend.xp
        nE, nFe = a.shape
        nFd = len(flavor_det)

        # Build V_{beta i} with detector as last axis -> (nE, nFe, nFd)
        if xp.ndim(V) == 2:
            Vb_T = xp.asarray(V[flavor_det, :]).T  # (nFe, nFd)
            Vb_T = xp.broadcast_to(Vb_T, (nFe, nFd))[None, ...]  # (1, nFe, nFd) -> (nE, nFe, nFd) via broadcast
        else:
            Vb = V[:, flavor_det, :]  # (nE, nFd, nFe)
            Vb_T = xp.swapaxes(Vb, 1, 2)  # (nE, nFe, nFd)

        # Per-eigenstate amplitudes: A_i(E, i, beta) = a_i(E) * V_{beta i}(E)
        A_i = a[:, :, None] * Vb_T  # (nE, nFe, nFd)

        # Incoherent contribution (per i): |A_i|^2
        P_incoh_i = xp.abs(A_i) ** 2  # (nE, nFe, nFd)

        # Total amplitude summed over i, then |.|^2
        amp_tot = xp.sum(A_i, axis=1)  # (nE, nFd)
        P_total_scalar = xp.abs(amp_tot) ** 2  # (nE, nFd)

        # Broadcast totals/interference so output dims are (nE, nFe, nFd)
        P_total = xp.broadcast_to(P_total_scalar[:, None, :], P_incoh_i.shape)
        P_incoh_sum = xp.sum(P_incoh_i, axis=1)  # (nE, nFd)
        P_int_scalar = P_total_scalar - P_incoh_sum  # (nE, nFd)
        P_int = xp.broadcast_to(P_int_scalar[:, None, :], P_incoh_i.shape)

        # print(f"P_total.shape = {P_total.shape}")
        # print(f"P_total = {P_total}")
        return {
            "total": self.backend.from_device(P_total),  # (nE, nFe, nFd)
            "incoherent": self.backend.from_device(P_incoh_i),  # (nE, nFe, nFd)
            "interference": self.backend.from_device(P_int),  # (nE, nFe, nFd)
        }

    def _probability(self, L, E, flavor_emit=None, flavor_det=None, antineutrino=False):
        return self._probability_split_adiabatic(
            L=L, E=E, flavor_emit=flavor_emit, flavor_det=flavor_det, antineutrino=antineutrino
        )["total"]

    def _generate_L_and_E_arrays(self, L_km, E_GeV):
        xp = self.backend.xp

        # ---------- normalize inputs ----------
        L_in = xp.asarray(L_km, dtype=self.backend.dtype_real)
        E_in = xp.asarray(E_GeV, dtype=self.backend.dtype_real)

        if xp.ndim(L_in) == 0:
            L_in = L_in.reshape(1)
        if xp.ndim(E_in) == 0:
            E_in = E_in.reshape(1)

        # ---------- enforce pairwise semantics ----------
        if xp.size(L_in) == 1 and xp.size(E_in) > 1:
            Lc = xp.broadcast_to(L_in, E_in.shape)
            Ec = E_in
        elif xp.size(E_in) == 1 and xp.size(L_in) > 1:
            Lc = L_in
            Ec = xp.broadcast_to(E_in, L_in.shape)
        else:
            if xp.size(L_in) != xp.size(E_in):
                raise ValueError(
                    f"Length mismatch: L_km has {xp.size(L_in)}, E_GeV has {xp.size(E_in)}. "
                    "They must match for pairwise propagation."
                )
            Lc, Ec = L_in, E_in

        center_shape = Lc.shape

        # ---------- prepare flattened arrays ----------
        E_flat = Ec.reshape(-1)
        L_flat = Lc.reshape(-1)

        return L_flat, E_flat

    def _probability_sampled(self, L_true_km, E_true_GeV, flavor_emit=None, flavor_det=None, antineutrino=False, L_sample_fct=None, E_sample_fct=None, nSamples=100):
        if L_sample_fct is None and E_sample_fct is None:
            raise AttributeError("At least one of L_sample_fct or L_sample_fct is required")

        xp = self.backend.xp
        L_true_km_tensor = xp.asarray(L_true_km) if not hasattr(L_true_km, 'shape') else L_true_km
        E_true_GeV_tensor = xp.asarray(E_true_GeV) if not hasattr(E_true_GeV, 'shape') else E_true_GeV
        E_true_GeV_tensor = xp.broadcast_to(E_true_GeV_tensor, L_true_km_tensor.shape) if L_true_km_tensor.ndim == 1 and E_true_GeV_tensor.ndim == 0 else E_true_GeV_tensor

        L_km = ...
        E_GeV = ...

        P = self._probability(L=L_km, E=E_GeV, flavor_emit=flavor_emit, flavor_det=flavor_det, antineutrino=antineutrino)

        #



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
        if xp.ndim(r_in) != 1 or xp.size(r_in) < 2:
            raise ValueError("r_km must be a 1D array with length ≥ 2.")

        # Work internally with increasing radius; remember if we reversed
        rev = r_in[0] > r_in[-1]
        r_path = r_in[::-1] if rev else r_in

        rho_np = profile.rho_gcm3(r_path)
        Ye_np = profile.Ye(r_path)
        H_list = []
        for k in range(xp.size(r_path)):
            Hk = self.hamiltonian.matter_constant(
                E_GeV, rho_gcm3=float(rho_np[k]), Ye=float(Ye_np[k]),
                antineutrino=antineutrino
            )
            H_list.append(Hk[0] if xp.ndim(Hk) == 3 else Hk)  # ensure (N,N)

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
        if xp.ndim(H) == 3: H = H[0]  # (N,N)

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
        if np.ndim(s) != 1 or np.size(s) < 2:
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
