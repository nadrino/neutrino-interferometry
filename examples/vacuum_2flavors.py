import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import VacuumOscillator
import nu_waves.utils.flavors as flavors

# sterile test
osc_amplitude = 0.1 # sin^2(2\theta)
angles = {(1, 2): np.arcsin(np.sqrt(osc_amplitude))/2}
pmns = Mixing(dim=2, mixing_angles=angles)
U_pmns = pmns.get_mixing_matrix()
print(np.round(U_pmns, 3))


# 1 eV^2
spec = Spectrum(n=2, m_lightest=0.)
spec.set_dm2({(2, 1): 1})
spec.summary()
m2_diag = np.diag(spec.get_m2())


osc = VacuumOscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2())

E_fixed = 5E-3
L_min, L_max = 1e-3, 20e-3
L_list = np.linspace(L_min, L_max, 200)
print(L_list)
P = osc.probability(
    L_km=L_list, E_GeV=E_fixed,
    alpha=flavors.electron,
    beta=flavors.electron, # muon could be sterile
    antineutrino=True
)
print(P)

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.figure(figsize=(6.5, 4.0))

plt.plot(L_list*1000, P, label=r"$P_{e e}$ disappearance", lw=2)
plt.plot(L_list*1000, [1]*len(L_list), "--", label="Total probability", lw=1.5)

plt.xlabel(r"$L_\nu$ [m]")
plt.ylabel(r"Probability")
plt.title(f"eV$^2$ sterile with $E_\\nu$ = {E_fixed*1000} MeV")
# plt.xlim(L_min, L_max)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()
