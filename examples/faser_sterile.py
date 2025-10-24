import numpy as np
import os
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style

# toggle GPU
# torch_backend = None
torch_backend = make_torch_backend(seed=0, use_complex64=True)

nBins_E = 100
nSamples_E = 1000
fixed_L = 0.2 # km
dm2_sterile = 1000 # eV2
E_GeV_list = np.logspace(2, 4, nBins_E)

# 3 flavors PMNS, PDG values (2025)
osc_amplitude = 0.5 # amplitude as ~RAA
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49), (4, 2): np.arcsin(np.sqrt(osc_amplitude))/2}
phases = {(1, 3): np.deg2rad(195)}

osc = Oscillator(
    mixing_matrix=Mixing(dim=4, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_list=Spectrum(n=4, m_lightest=0., dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428, (4, 1): dm2_sterile}).get_m2(),
    backend=torch_backend
)
xp = osc.backend.xp

def energy_sampler_sqrt(E_center, n, a=10):
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
P_mumu = osc.probability(
    L_km=fixed_L, E_GeV=E_GeV_list,
    flavor_emit=flavors.muon,
    flavor_det=flavors.muon, # muon could be sterile
    antineutrino=False
)

print(P_mumu)

plt.figure(figsize=(6.5, 4.0), dpi=150)

plt.plot(E_GeV_list*1000, [1]*len(E_GeV_list), "--", label=r"No oscillation", lw=2)
plt.plot(E_GeV_list*1000, P_mumu, label=r"$P_{\mu\mu}$ disappearance", lw=2)

plt.xlabel(r"$E_\nu$ [TeV]")
plt.ylabel(r"Probability")
plt.title(f"{dm2_sterile} eV$^2$ sterile with $L_\\nu$ = {fixed_L*1000} m")
plt.xscale("log")
plt.xlim(left=E_GeV_list[0]*1000, right=E_GeV_list[-1]*1000)
plt.ylim(0, 1.1)
plt.legend()

plt.grid(True, which="both", alpha=0.3, lw=0.4, ls="--")
plt.minorticks_on()

plt.tight_layout()

plt.tight_layout()
# plt.savefig("../figures/sterile_raa_plot.jpg", dpi=150)  if not os.environ.get("CI") else None
plt.show()



