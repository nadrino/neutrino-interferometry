import numpy as np
import os
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style


nBins_L = 200
nSamples_E = 1000
fixed_E = 0.003 # GeV
L_km_list = np.logspace(-2, 2.5, nBins_L)

# 3 flavors PMNS, PDG values (2025)
osc_amplitude = 0.06 # amplitude as ~RAA
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49), (4, 1): np.arcsin(np.sqrt(osc_amplitude))/2}
phases = {(1, 3): np.deg2rad(195)}

osc = Oscillator(
    mixing_matrix=Mixing(n_neutrinos=4, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_list=Spectrum(n_neutrinos=4, m_lightest=0., dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428, (4, 1): 1}).get_m2()
)
xp = osc.backend.xp


def energy_sampler_sqrt(E_center, n, a=0.004):
    """
    Gaussian energy smearing with sigma(E) = a * sqrt(E).
    - E_center: scalar/array/grid of energies [GeV]
    - n: number of samples
    - a: resolution scale [GeV**0.5] (e.g. a=0.08 ⇒ 8% at 1 GeV)

    Returns: samples with shape E_center.shape + (n,)
    """
    E = xp.asarray(E_center, dtype=float)
    # avoid sqrt of negatives; also avoids zero-variance at E=0
    E_safe = xp.maximum(E, 1e-12)
    sigma = a * xp.sqrt(E_safe)
    out = xp.normal(loc=E[..., None], scale=sigma[..., None], size=E.shape + (n,))
    return out


osc.energy_sampler = energy_sampler_sqrt
osc.n_samples = nSamples_E
P_ee = osc.probability(
    L_km=L_km_list, E_GeV=fixed_E,
    flavor_emit=flavors.electron,
    flavor_det=flavors.electron, # muon could be sterile
    antineutrino=True
)

angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
osc.set_parameters(
    mixing_matrix=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_list=Spectrum(n_neutrinos=3, m_lightest=0., dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428}).get_m2()
)

P_ee_orig = osc.probability(
    L_km=L_km_list, E_GeV=fixed_E,
    flavor_emit=flavors.electron,
    flavor_det=flavors.electron, # muon could be sterile
    antineutrino=True
)


plt.figure(figsize=(6.5, 4.0), dpi=150)

plt.plot(L_km_list*1000, [1]*len(L_km_list), "--", label=r"No oscillation", lw=2)
plt.plot(L_km_list*1000, P_ee_orig, label=r"$P_{e e}$ disappearance", lw=2)
plt.plot(L_km_list*1000, P_ee, label=r"$P_{e e}$ disappearance (with sterile)", lw=2)

plt.xlabel(r"$L_\nu$ [m]")
plt.ylabel(r"Probability")
plt.title(f"eV$^2$ sterile with $E_\\nu$ = {fixed_E*1000} MeV")
plt.xscale("log")
plt.xlim(left=L_km_list[0]*1000, right=L_km_list[-1]*1000)
plt.ylim(0, 1.1)
plt.legend()

plt.grid(True, which="both", alpha=0.3, lw=0.4, ls="--")
plt.minorticks_on()

plt.tight_layout()

plt.tight_layout()
plt.savefig("../figures/sterile_raa_plot.jpg", dpi=150)  if not os.environ.get("CI") else None
plt.show()



