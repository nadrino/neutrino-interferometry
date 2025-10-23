from nu_waves.hamiltonian.base import Hamiltonian
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.backends.global_backend import get_backend

class VacuumHamiltonian(Hamiltonian):
    def __init__(self, U_flavor_from_mass, m2_eV2):
        nF = U_flavor_from_mass.shape[0]

        get_backend().xp

        super().__init__(xp=xp, n_flavors=nF)
        self.U = xp.asarray(U_flavor_from_mass, dtype=xp)     # (nF,nF)
        self.Ud = xp.conjugate(self.U).T                          # U†
        self.m2 = xp.asarray(m2_eV2, dtype=cdtype)                # (nF,)

    def propagate_state(self, psi: WaveFunction, E_GeV, L_km) -> WaveFunction:
        xp = self.xp
        E = xp.asarray(E_GeV, dtype=self.cdtype)                  # (nE,)
        nE = E.shape[0]

        # phases φ_i = 1.267 * Δm_i^2[eV^2] * L[km] / E[GeV]
        phi = 1.267 * (L_km / E)[:, None] * self.m2[None, :]      # (nE,nF)
        D = xp.exp(-1j * phi)                                     # (nE,nF)

        # Rotate ψ to mass basis only if needed
        if psi.basis == Basis.MASS:
            psi_m = psi.values                                     # (nE,nF)
        elif psi.basis == Basis.FLAVOR:
            psi_m = (self.Ud[None, :, :] @ psi.values[..., None])[..., 0]
        else:
            raise ValueError("VacuumHamiltonian: unsupported input basis for ψ.")

        # Diagonal propagation in mass basis
        psi_m = psi_m * D

        # Rotate back to flavor (propagators in your code downstream expect flavor)
        psi_f = (self.U[None, :, :] @ psi_m[..., None])[..., 0]
        return psi.copy_like(values=psi_f, basis=Basis.FLAVOR)

    # If you still want a flavor-basis S for other callers:
    def propagator(self, E_GeV, L_km):
        xp = self.xp
        E = xp.asarray(E_GeV, dtype=self.cdtype)
        phi = 1.267 * (L_km / E)[:, None] * self.m2[None, :]
        D = xp.exp(-1j * phi)                                     # (nE,nF)
        U, Ud = self.U, self.Ud
        S = (U[None, :, :] * D[:, None, :]) @ Ud[None, :, :]
        return S.astype(self.cdtype)

