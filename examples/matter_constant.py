import numpy as np
import os
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style

# backend = None
backend = make_torch_backend(
    force_device="cpu"
)


# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.get_mixing_matrix()

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})

# oscillator
osc = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2(), backend=backend)

# calculate without matter effects
osc.use_vacuum()
P_vac = osc.probability(L_km=295, E_GeV=np.linspace(0.2,2,50), flavor_emit=flavors.muon, flavor_det=flavors.electron)

# compare with density = 0
osc.set_constant_density(rho_gcm3=0.0, Ye=0.5)
P_zero = osc.probability(L_km=295, E_GeV=np.linspace(0.2,2,50), flavor_emit=flavors.muon, flavor_det=flavors.electron)

# should be equal
np.testing.assert_allclose(P_vac, P_zero, atol=1e-12)


# --- DUNE-like configuration ---
L_km = 1300.0                  # Fermilab → SURF
rho_gcm3, Ye = 2.8, 0.5        # average crust density and electron fraction
E = np.linspace(0.2, 5.0, 600) # GeV

# --- Vacuum probabilities ---
osc.use_vacuum()
P_mue_vac = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=0)   # νμ→νe

# --- Constant-density matter probabilities ---
osc.set_constant_density(rho_gcm3=rho_gcm3, Ye=Ye)
P_mue_matt = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=0)

# --- (optional) antineutrinos for comparison ---
P_muebar_matt = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=0, antineutrino=True)

# --- Plot ---
plt.figure(figsize=(7,4.2))
plt.plot(E, P_mue_vac,  label=r"$\nu_\mu\!\to\!\nu_e$ (vac)", lw=2)
plt.plot(E, P_mue_matt, label=r"$\nu_\mu\!\to\!\nu_e$ (matter)", lw=2)
plt.plot(E, P_muebar_matt, ":", label=r"$\bar\nu_\mu\!\to\!\bar\nu_e$ (matter)", lw=2)

plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel("Probability")
plt.title("DUNE-like oscillation, L=1300 km (vacuum vs matter)")
plt.xlim(E.min(), E.max())
plt.ylim(0, 0.3)
plt.legend(ncol=2, frameon=False)
plt.tight_layout()
plt.show()

# inverted ordering?
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): -0.0024428})

# oscillator
osc = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2())

# calculate without matter effects
osc.set_constant_density(rho_gcm3=rho_gcm3, Ye=Ye)
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

