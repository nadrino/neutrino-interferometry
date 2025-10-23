import numpy as np
from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.backends import global_backend as backend
from nu_waves.matter.profile import MatterProfile
from nu_waves.utils.units import KM_TO_EVINV
from nu_waves.utils.units import GEV_TO_EV
from nu_waves.state.wave_function import WaveFunction, Basis


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
        Generate the initial neutrino state |ν_flavor_emit⟩ for each energy.

        Parameters
        ----------
        flavor_emit : int
            Index of the emitted flavor (0 = ν_e, 1 = ν_μ, 2 = ν_τ, ...)
        E : xp.ndarray
            Energies of the neutrinos (shape (nE,))
        antineutrino : bool, optional
            If True, use the CP-conjugate state (same flavor vector, but you may
            later apply conjugate mixing matrices). Default: False.

        Returns
        -------
        Wavefunction
            Object containing the (nE, nF) complex array in the FLAVOR basis.
        """
        xp = self.backend.xp
        nE = E.shape[0]
        nF = self.hamiltonian.n_flavors

        # Create zero-filled wavefunction array
        psi = xp.zeros((nE, nF), dtype=self.backend.ctype)

        # Set the emitted flavor amplitude to 1
        psi[:, flavor_emit] = 1.0

        # Create the holder
        state = WaveFunction(
            current_basis=Basis.FLAVOR,
            values=psi,
            antineutrino=antineutrino
        )

        return state

    def _propagate_state(self, psi: WaveFunction, L, E, antineutrino=False):
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
                    evals, evecs = self.backend.linalg.eigh(HL)  # HL = V diag(e) V^†
                    phases = xp.exp((-1j) * evals)  # (nE,nF)
                    # Torch supports only transpose(dim0, dim1)
                    # NumPy accepts transpose with any number of dims
                    VcT = xp.conjugate(evecs)
                    if hasattr(VcT, "permute"):  # torch.Tensor
                        VcT = VcT.permute(0, 2, 1)  # (nE, nF, nF)
                    else:
                        VcT = VcT.transpose(0, 2, 1)  # numpy
                    S = (evecs * phases[:, xp.newaxis, :]) @ VcT
                    # S = (evecs * phases[:, xp.newaxis, :]) @ xp.conjugate(evecs).transpose(0, 2, 1) # numpy
                    # S = (evecs * phases[:, xp.newaxis, :]) @ xp.conjugate(xp.transpose_batched(evecs)) # torch
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
                        evals, evecs = self.backend.linalg.eigh(HLk)
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
                U = xp.asarray(self.hamiltonian.U, dtype=dtype_c)
                m2 = xp.asarray(self.hamiltonian.m2_diag, dtype=dtype_r)
                phase = 0.5 * (L / E)[:, None] * m2[None, :]
                phases = xp.exp((-1j) * phase)
                Uc = xp.conjugate(U).T
                U_phase = U[xp.newaxis, :, :] * phases[:, xp.newaxis, :]
                S = U_phase @ Uc  # (nE,nF,nF)

        # ---------- apply initial state(s): psi_out(b,i) = sum_j S(b,i,j) psi(b,j) ----------
        psi = (S[:, None, ...] @ psi[..., None])[..., 0]

        return psi

    def _project_state(self, psi, E=None, antineutrino=False):
        """
        psi: (nE, ..., nF) in flavour basis
        returns:
          a: (nE, ..., nF) amplitudes in mass/matter basis
          V: (nF,nF) in vacuum or (nE,nF,nF) in matter
        """
        xp = self.backend.xp
        dtype_c = self.backend.dtype_complex

        if not self._use_matter:
            V = xp.asarray(self.hamiltonian.U, dtype=dtype_c)  # (nF,nF)
            a = psi @ xp.conjugate(V)
            return a, V

        assert E is not None
        if getattr(self, "_matter_profile", None) is None:
            rho, Ye = self._matter_args
        else:
            layer = self._matter_profile.layers[-1]  # detection layer
            rho, Ye = layer.rho_gcm3, layer.Ye
        H = self.hamiltonian.matter_constant(E, rho_gcm3=rho, Ye=Ye, antineutrino=antineutrino)
        _, V = self.backend.linalg.eigh(H)
        a = xp.matmul(psi, xp.conjugate(V))
        return a, V

    def _probability_split_adiabatic(self, L, E, flavor_emit=None, flavor_det=None, antineutrino=False):
        xp = self.backend.xp

        psi0 = self._generate_initial_state(flavor_emit=flavor_emit, E=E, antineutrino=antineutrino)
        psi = self._propagate_state(psi=psi0, L=L, E=E, antineutrino=antineutrino)
        a, V = self._project_state(
            psi=psi,
            E=E if self._use_matter else None,
            antineutrino=antineutrino if self._use_matter else None
        )

        # Detector projectors and amplitude sum over mass index j
        if xp.ndim(V) == 2:  # vacuum
            A = a @ V[flavor_det, :].T #
        else:  # matter
            VbT = xp.swapaxes(V[:, flavor_det, :], -1, -2)  # (nE, nF, nFd)
            A = xp.matmul(a, VbT)  # (nE, nFe, nFd)

        P_total = xp.abs(A) ** 2  # (nE, nFe, nFd)
        return {
            "total": P_total,
            # Optional detailed pieces:
            # "incoherent":   self.backend.from_device(xp.einsum("bkj,bej->bek", xp.abs(Vb)**2, xp.abs(a)**2)),
            # "interference": self.backend.from_device(P_total - previous_line),
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
        E_flat = Ec.reshape(-1) * GEV_TO_EV
        L_flat = Lc.reshape(-1) * KM_TO_EVINV

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
