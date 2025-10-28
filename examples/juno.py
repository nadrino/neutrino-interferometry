import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import vacuum
from nu_waves.globals.backend import Backend
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.sources.reactor_flux import ReactorSpectrum, FissionFractions
from nu_waves.interactions.ibd_cross_section import IBDCrossSection
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style

E = np.linspace(1.8, 9.5, 4000)
E_GeV = E*1E-3

# 1. Build non-oscillated emitted flux × σ_IBD
flux_model = ReactorSpectrum(FissionFractions(0.55, 0.30, 0.07, 0.08))
ibd = IBDCrossSection()
flux = flux_model.build_flux(E)
det_flux = ibd.weighted_flux(E, flux)  # this is your PDF before oscillations

angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}
h = vacuum.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, m_lightest=0, dm2=dm2),
    antineutrino=True
)
osc = Oscillator(hamiltonian=h)

spectrum_NoOsc = det_flux

Pee_NO = osc.probability(L_km=52.5, E_GeV=E_GeV, flavor_emit=flavors.electron, flavor_det=flavors.electron)
spectrum_NO = det_flux * Pee_NO

# inverted ordering
osc.hamiltonian.spectrum.set_dm2({(3, 2): -dm2[(3, 2)]})
Pee_IO = osc.probability(L_km=52.5, E_GeV=E_GeV, flavor_emit=flavors.electron, flavor_det=flavors.electron)
spectrum_IO = det_flux * Pee_IO

# plot the results
plt.figure(figsize=(7,4), dpi=150)
# plt.plot(E, spectrum_NoOsc, label="No oscillations")
plt.plot(E, spectrum_NO, label="Detected spectrum (NO)")
plt.plot(E, spectrum_IO, "--", label="Inverted ordering (IO)")


plt.xlabel("Neutrino energy [MeV]")
plt.ylabel("Arbitrary units")
plt.title("Reactor antineutrino spectrum at JUNO baseline")
plt.grid(True, alpha=0.3)
plt.xlim(left=1.6, right=9)
plt.legend(loc="best")
plt.tight_layout()
plt.show()




