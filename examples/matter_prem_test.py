# examples/oscillogram_prem_refined.py
import numpy as np
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import VacuumOscillator
from nu_waves.matter.prem import PREMModel
from nu_waves.backends import make_torch_mps_backend

# toggle for CPU/GPU
# torch_backend = None
torch_backend = make_torch_mps_backend(seed=0, use_complex64=True)

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.get_mixing_matrix()

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})


osc = VacuumOscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2(), backend=torch_backend)

L = 8000.0
E = np.linspace(1, 10, 400)
P_NO = osc.probability(L_km=L, E_GeV=E, alpha=1, beta=0)              # neutrinos, NO
P_IO = osc.probability(L_km=L, E_GeV=E, alpha=1, beta=0, antineutrino=True)  # ν̄ in IO has the resonance

P_NO = torch_backend.from_device(P_NO)
P_IO = torch_backend.from_device(P_IO)
print("P_mu->e max (NO, ν):", P_NO.max(), "  P_mu->e max (IO, ν̄):", P_IO.max())

E_GeV = np.logspace(-1, 2, 400)     # x
cosz  = np.linspace(-1.0, 1.0, 240)     # y (upgoing)
prem  = PREMModel()

P_mue  = np.zeros((cosz.size, E_GeV.size))
P_mumu = np.zeros_like(P_mue)

# choose one:
SCHEME = "prem_layers"      # exact PREM shells
# SCHEME = "hist_density"   # fine histogram of density along path

for iy, cz in enumerate(cosz):
    prof = prem.profile_from_coszen(cz, scheme=SCHEME, n_bins=1200, nbins_density=36, merge_tol=0.0)
    osc.set_layered_profile(prof)

    # total path length is sum of absolute segments
    L_tot = float(sum(layer.weight for layer in prof.layers))

    P_mue_i  = osc.probability(L_km=L_tot, E_GeV=E_GeV, alpha=1, beta=0)
    P_mumu_i = osc.probability(L_km=L_tot, E_GeV=E_GeV, alpha=1, beta=1)

    if torch_backend is not None:
        P_mue[iy] = torch_backend.from_device(P_mue_i)
        P_mumu[iy] = torch_backend.from_device(P_mumu_i)
    else:
        P_mue[iy] = P_mue_i
        P_mumu[iy] = P_mumu_i


def draw(ax, X, Y, Z, title):
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   vmin=0.0, vmax=1.0, interpolation="nearest", cmap='inferno')
    E_edges = np.geomspace(0.1, 100.0, len(E_GeV) + 1)
    E_centers = np.sqrt(E_edges[:-1] * E_edges[1:])  # compute P on centers
    CZ_edges = np.linspace(-1.0, 1.0, len(cosz) + 1)
    pc = ax.pcolormesh(E_edges, CZ_edges, Z, vmin=0, vmax=1, shading="auto")
    ax.set_xscale('log')
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(r"$\cos\theta_z$")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Probability")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
draw(axs[0], E_GeV, cosz, P_mue,  r"$P(\nu_\mu\to\nu_e)$ — PREM refined")
draw(axs[1], E_GeV, cosz, P_mumu, r"$P(\nu_\mu\to\nu_\mu)$ — PREM refined")
plt.show()
