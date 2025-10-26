import numpy as np
import os
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import vacuum
from nu_waves.hamiltonian import matter
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.globals.backend import Backend
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style

import torch
Backend.set_api(torch, device='mps')

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
# Masses, normal ordering
dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}

h_vacuum = vacuum.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, m_lightest=0, dm2=dm2),
    antineutrino=False
)

h_matter = matter.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, m_lightest=0, dm2=dm2),
    antineutrino=False
)
h_matter.set_constant_density(rho_in_g_per_cm3=0.)


# calculate without matter effects
osc = Oscillator(hamiltonian=h_vacuum)
P_vacuum = osc.probability(L_km=295, E_GeV=np.linspace(0.2, 2, 50), flavor_emit=flavors.muon, flavor_det=flavors.electron)

# compare with density = 0
osc.hamiltonian = h_matter
P_zero_density = osc.probability(L_km=295, E_GeV=np.linspace(0.2, 2, 50), flavor_emit=flavors.muon, flavor_det=flavors.electron)

# should be equal
delta = np.abs(P_vacuum - P_zero_density)
print(delta)
np.testing.assert_allclose(delta, 0, atol=1e-7)


# --- DUNE-like configuration ---
L_km = 1300.0                  # Fermilab → SURF
rho_gcm3, Ye = 2.8, 0.5        # average crust density and electron fraction
E = np.linspace(0.2, 5.0, 600) # GeV

# --- Vacuum probabilities ---
# h_matter.set_constant_density(rho_in_g_per_cm3=rho_gcm3, Ye=0.5)
osc.hamiltonian = h_vacuum
P_mue_vac = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=0)   # νμ→νe

# --- Constant-density matter probabilities ---
osc.hamiltonian = h_matter
h_matter.set_constant_density(rho_in_g_per_cm3=rho_gcm3, Ye=0.5)
P_mue_matt = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=0)

# --- (optional) antineutrinos for comparison ---
h_matter.set_antineutrino(True)
P_muebar_matt = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=0)

# --- Plot ---
plt.figure(figsize=(7,4.2))
plt.plot(E, P_mue_vac,  label=r"$\nu_\mu\!\to\!\nu_e$ (vacuum)", lw=2)
plt.plot(E, P_mue_matt, label=r"$\nu_\mu\!\to\!\nu_e$ (matter)", lw=2)
plt.plot(E, P_muebar_matt, ":", label=r"$\bar\nu_\mu\!\to\!\bar\nu_e$ (matter)", lw=2)

plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel("Probability")
plt.title("DUNE-like oscillation, L=1300 km (vacuum vs matter)")
plt.xlim(E.min(), E.max())
plt.ylim(0, 0.3)
plt.legend(ncol=2, frameon=False)
plt.tight_layout()
plt.savefig("../figures/matter_vacuum_vs_matter.jpg", dpi=150)  if not os.environ.get("CI") else None
plt.show()


# inverted ordering
h_matter.spectrum.set_dm2({(2, 1): 7.42e-5, (3, 2): -0.0024428})
h_matter.set_antineutrino(False)
P_mue_matt_inv = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=0)

plt.figure(figsize=(7,4.2))
plt.plot(E, P_mue_matt, label=r"$\nu_\mu\!\to\!\nu_e$ (matter) NO", lw=2)
plt.plot(E, P_mue_matt_inv, label=r"$\nu_\mu\!\to\!\nu_e$ (matter) IO", lw=2)

plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel("Probability")
plt.title("DUNE-like oscillation, L=1300 km (vacuum vs matter)")
plt.xlim(E.min(), E.max())
plt.ylim(0, 0.3)
plt.legend(ncol=2, frameon=False)
plt.tight_layout()

plt.savefig("../figures/matter_constant_test.jpg", dpi=150)  if not os.environ.get("CI") else None
plt.show()

