import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.backends import make_torch_mps_backend
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.get_mixing_matrix()

# Masses, normal ordering
spec = Spectrum(n=3)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})

# toggle for CPU/GPU
# torch_backend = None
torch_backend = make_torch_mps_backend(seed=0, use_complex64=True)

# oscillator
osc = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2(), backend=torch_backend)

# --- Grid definition (x: energy, y: baseline) ---
E_vals = np.linspace(0.2, 5.0, 1000)     # GeV  (x-axis)
L_vals = np.linspace(1000, 2000.0, 1000) # km   (y-axis)

# --- Choose vacuum or matter ---
# osc.use_vacuum()  # vacuum
# For constant density (crust-like), uncomment:
osc.set_constant_density(rho_gcm3=2.8, Ye=0.5)

t0 = time.perf_counter()

# --- Compute probabilities in grid mode (shape -> (nL, nE) after selection) ---
# alpha=1 (muon), beta=0 (electron) → appearance
P_mue = osc.probability(L_km=L_vals, E_GeV=E_vals, alpha=1, beta=0)   # (nL, nE)
# alpha=1 (muon), beta=1 (muon) → disappearance
P_mumu = osc.probability(L_km=L_vals, E_GeV=E_vals, alpha=1, beta=1)  # (nL, nE)

t1 = time.perf_counter()
print(f"Computation time: {t1 - t0:.3f} s")

# --- Plot helper ---
def plot_oscillogram(ax, E, L, P, title):
    # imshow expects array indexed as [y, x] = [L, E]
    im = ax.imshow(
        P + 1e-10,  # avoid log(0)
        origin="lower",
        aspect="auto",
        extent=[E.min(), E.max(), L.min(), L.max()],
        norm=LogNorm(vmin=1e-3, vmax=1.0),
        interpolation="nearest",
        cmap='inferno'
    )
    ax.axhline(y=1300, color='white', linestyle='--', alpha=0.5)
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(r"$L$ [km]")
    
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Probability")

# --- Draw ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
plot_oscillogram(axes[0], E_vals, L_vals, P_mue,  r"$P(\nu_\mu\to\nu_e)$")
plot_oscillogram(axes[1], E_vals, L_vals, P_mumu, r"$P(\nu_\mu\to\nu_\mu)$")


# plt.savefig("../figures/vacuum_2d_pmns.pdf") # too heavy
plt.savefig("../figures/vacuum_2d_pmns.jpg", dpi=150)
plt.show()

