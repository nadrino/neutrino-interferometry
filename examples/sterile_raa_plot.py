import numpy as np
import os
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import vacuum
from nu_waves.propagation.new_oscillator import Oscillator
from nu_waves.globals.backend import Backend
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style

import torch
Backend.set_api(torch, device='mps')


nBins_L = 200
E_res = 0.05
nSamples_E = 10000
fixed_E = 0.003 # GeV
L_km_list = np.logspace(-2, 2.5, nBins_L)

# 3 flavors PMNS, PDG values (2025)
osc_amplitude = 0.06 # amplitude as ~RAA
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49), (4, 1): np.arcsin(np.sqrt(osc_amplitude))/2}
phases = {(1, 3): np.deg2rad(195)}

h = vacuum.Hamiltonian(
    mixing_matrix=Mixing(n_neutrinos=4, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_array=Spectrum(n_neutrinos=4, m_lightest=0., dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428, (4, 1): 1}).get_m2()
)
osc = Oscillator(hamiltonian=h)

# Example: fractional Gaussian energy resolution and fixed baseline blur
def gaussian_E_sampler(E, n_samples, sigma_rel=E_res):
    # E: (nE,)
    xp = Backend.xp()
    noise = sigma_rel * xp.abs(E)[:, None] * xp.asarray(
        xp.random.standard_normal((E.shape[0], n_samples)), dtype=Backend.real_dtype()
    )
    out = E[:, None] + noise
    return xp.maximum(out, xp.asarray(1e-12, dtype=Backend.real_dtype()))


P_ee = osc.probability_sampled(
    L_km=L_km_list, E_GeV=fixed_E,
    flavor_emit=flavors.electron,
    flavor_det=flavors.electron,
    antineutrino=True,
    n_samples=nSamples_E,
    E_sample_fct=gaussian_E_sampler,
)

# now switching back to 3 flavors
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
h = vacuum.Hamiltonian(
    mixing_matrix=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_array=Spectrum(n_neutrinos=3, m_lightest=0., dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428}).get_m2()
)
osc.hamiltonian = h




P_ee_orig = osc.probability_sampled(
    L_km=L_km_list, E_GeV=fixed_E,
    flavor_emit=flavors.electron,
    flavor_det=flavors.electron,
    antineutrino=True,
    n_samples=nSamples_E,
    E_sample_fct=gaussian_E_sampler,
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



