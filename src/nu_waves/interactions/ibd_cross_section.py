import numpy as np

class IBDCrossSection:
    """
    Compute the inverse beta decay (IBD) cross-section σ(Eν).
    Default: Vogel–Beacom approximate analytical form.
    """

    def __init__(self):
        # physical constants in MeV
        self.me = 0.511      # electron mass
        self.delta = 1.293   # neutron–proton mass difference
        self.threshold = 1.806

    def sigma(self, E_nu):
        """
        Approximate total cross-section (shape only, arbitrary units).
        σ ≈ 0.0952 × (Ee p_e), Ee = Eν − Δ, p_e = √(Ee² − me²)
        """
        E_nu = np.asarray(E_nu)
        Ee = E_nu - self.delta
        Ee = np.clip(Ee, 0, None)
        pe = np.sqrt(np.clip(Ee**2 - self.me**2, 0, None))
        sigma = 0.0952 * Ee * pe
        sigma[E_nu < self.threshold] = 0.0
        return sigma

    def weighted_flux(self, E_nu, flux):
        """Return flux × σ(Eν), the detected-energy PDF (before oscillations)."""
        return flux * self.sigma(E_nu)
