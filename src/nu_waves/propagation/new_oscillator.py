from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.utils.units import GEV_TO_EV, KM_TO_EVINV


class Oscillator:

    def __init__(self, hamiltonian: Hamiltonian):
        self.hamiltonian = hamiltonian

    def probability(self, L_km, E_GeV, flavor_emit=None, flavor_det=None, antineutrino=False):
        L, E = self._generate_L_and_E_arrays(L_km, E_GeV)
        flavor_emit = self._format_flavor_arg(flavor_emit)
        flavor_det = self._format_flavor_arg(flavor_det)
        out = self._probability(L=L, E=E, flavor_emit=flavor_emit, flavor_det=flavor_det, antineutrino=antineutrino)
        return Backend.from_device(self._squeeze_array(out))

    def generate_initial_state(self, flavor_emit, E_GeV, antineutrino=False):
        E = Backend.xp().asarray(E_GeV) * GEV_TO_EV
        return Backend.from_device(
            self._generate_initial_state(flavor_emit=flavor_emit, E=E, antineutrino=antineutrino)
        )

    def _probability(self, L, E, flavor_emit, flavor_det, antineutrino):
        psi = self._generate_initial_state(flavor_emit=flavor_emit, E=E, antineutrino=antineutrino)
        self.hamiltonian.propagate_state(psi=psi, L=L, E=E) # return the state in flavor basis

        nF = self.hamiltonian.n_flavors
        nFd = len(flavor_det)
        flavor_det_vecs = Backend.xp().zeros((nFd, nF), dtype=Backend.complex_dtype())
        flavor_det_vecs[Backend.xp().arange(nFd), flavor_det] = 1.0  # (nFd, nF)
        amp = psi.values @ Backend.xp().conjugate(flavor_det_vecs.T)
        return Backend.xp().abs(amp)**2

    def _generate_initial_state(self, flavor_emit, E, antineutrino) -> WaveFunction:
        xp = Backend.xp()
        nE = E.shape[0]
        nF = self.hamiltonian.n_flavors
        nFe = len(flavor_emit)

        # Create zero-filled wavefunction array
        psi = xp.zeros((nE, nFe, nF), dtype=Backend.complex_dtype())

        # Set the emitted flavor amplitude to 1
        psi[:, xp.arange(nFe), flavor_emit] = 1.0

        # Create the holder
        return WaveFunction(
            current_basis=Basis.FLAVOR,
            values=psi,
            antineutrino=antineutrino
        )

    def _generate_L_and_E_arrays(self, L_km, E_GeV):
        # ---------- normalize inputs ----------
        xp = Backend.xp()

        L_in = xp.asarray(L_km, dtype=Backend.real_dtype())
        E_in = xp.asarray(E_GeV, dtype=Backend.real_dtype())

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

    def _format_flavor_arg(self, arg):
        """
        Normalize to list[int].
        Allowed: None → full range [0..n_flavors-1], int, list[int].
        Disallowed: anything else.
        """
        xp = Backend.xp()
        if arg is None:
            return list(range(int(self.hamiltonian.n_flavors)))

        if isinstance(arg, int):
            out = [int(arg)]
        elif isinstance(arg, list) and all(isinstance(x, int) for x in arg):
            out = [int(x) for x in arg]
        else:
            raise TypeError(f"Flavor arg must be None, int, or list of int.")

        return out

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