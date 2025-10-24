import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.matter.prem import PREMModel
from nu_waves.matter.profile import MatterProfile
import nu_waves.utils.style

# toggle for CPU/GPU
# torch_backend = None
torch_backend = make_torch_backend(seed=0, use_complex64=True)

# choose one:
# SCHEME = "prem_layers"      # exact PREM shells
SCHEME = "hist_density"   # fine histogram of density along path


# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.get_mixing_matrix()

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})

osc = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2(), backend=torch_backend)

# sanity test for layer ordering
prof = MatterProfile.from_segments(
    rho_gcm3=[2.8, 11.0], Ye=[0.5, 0.467], lengths_km=[3000.0, 2000.0]
)
osc.set_layered_profile(prof)
E = np.linspace(2,8,200)
P_fwd = osc.probability(L_km=sum(l.weight for l in prof.layers), E_GeV=E, flavor_emit=1, flavor_det=0)

# reverse the profile
prof_rev = MatterProfile.from_segments(
    rho_gcm3=[11.0, 2.8], Ye=[0.467, 0.5], lengths_km=[2000.0, 3000.0]
)
osc.set_layered_profile(prof_rev)
P_rev = osc.probability(L_km=sum(l.weight for l in prof_rev.layers), E_GeV=E, flavor_emit=1, flavor_det=0)

print("max |ΔP| (fwd vs rev) =", np.max(np.abs(P_fwd - P_rev)))  # should be >> 1e-3

L = 8000.0
E = np.linspace(1, 10, 400)
P_NO = osc.probability(L_km=L, E_GeV=E, flavor_emit=1, flavor_det=0)              # neutrinos, NO
P_IO = osc.probability(L_km=L, E_GeV=E, flavor_emit=1, flavor_det=0, antineutrino=True)  # ν̄ in IO has the resonance

print("P_mu->e max (NO, ν):", P_NO.max(), "  P_mu->e max (IO, ν̄):", P_IO.max())

E_GeV = np.logspace(-1, 2, 400)     # x
cosz  = np.linspace(-1.0, 1.0, 200)     # y (upgoing)
prem  = PREMModel()

xp = osc.backend.xp
def e_smear(E_center, n, a=0.01):
    E = xp.asarray(E_center, dtype=float)
    sigma = a * E
    out = xp.normal(loc=E[..., None], scale=sigma[..., None], size=E.shape + (n,))
    return out

osc.energy_sampler = e_smear
osc.n_samples = 50

# test for thickness
prof = prem.profile_from_coszen(+0.3, h_atm_km=15.0)
print("L_tot(downgoing) =", sum(L.weight for L in prof.layers))  # ≈ 15 km

for cz in (-1e-3, +1e-3):
    prof = prem.profile_from_coszen(cz, h_atm_km=15.0)
    print(cz, "L_atm =",
          sum(L.weight for L in prof.layers if L.rho_gcm3==0.0))

prof1 = prem.profile_from_coszen(-1, scheme="prem_layers")
prof2 = prem.profile_from_coszen(-1, scheme="hist_density", n_bins=4000, nbins_density=60)
osc.set_layered_profile(prof1); P1 = osc.probability(L_km=sum(l.weight for l in prof1.layers), E_GeV=E, flavor_emit=1, flavor_det=0)
osc.set_layered_profile(prof2); P2 = osc.probability(L_km=sum(l.weight for l in prof2.layers), E_GeV=E, flavor_emit=1, flavor_det=0)

print("max |ΔP| prem vs hist =", np.max(np.abs(P1-P2)))

# --- arrays to hold all 4 panels ---
P_mue      = np.zeros((len(cosz), len(E_GeV)))
P_mumu     = np.zeros_like(P_mue)
P_mue_bar  = np.zeros_like(P_mue)
P_mumu_bar = np.zeros_like(P_mue)

for iy, cz in tqdm(enumerate(cosz), total=len(cosz)):
    # 1) build PREM profile for this cos(zenith)
    prof = prem.profile_from_coszen(
        cz, scheme=SCHEME,
        n_bins=1200, nbins_density=36, merge_tol=0.0,  # keep your knobs
        h_atm_km=15.0                                   # thin atmosphere (your choice)
    )
    osc.set_layered_profile(prof)

    # 2) total baseline for this row (sum of absolute segments)
    L_tot = float(sum(layer.weight for layer in prof.layers))

    # 3) compute ν and ν̄ for the two channels
    P_mue_i      = osc.probability(L_km=L_tot, E_GeV=E_GeV, flavor_emit=1, flavor_det=0, antineutrino=False)
    P_mumu_i     = osc.probability(L_km=L_tot, E_GeV=E_GeV, flavor_emit=1, flavor_det=1, antineutrino=False)
    P_mue_bar_i  = osc.probability(L_km=L_tot, E_GeV=E_GeV, flavor_emit=1, flavor_det=0, antineutrino=True)
    P_mumu_bar_i = osc.probability(L_km=L_tot, E_GeV=E_GeV, flavor_emit=1, flavor_det=1, antineutrino=True)

    P_mue[iy], P_mumu[iy] = P_mue_i, P_mumu_i
    P_mue_bar[iy], P_mumu_bar[iy] = P_mue_bar_i, P_mumu_bar_i

# (optional) synchronize GPU timing before plotting
try:
    import torch
    if torch.backends.mps.is_available(): torch.mps.synchronize()
    elif torch.cuda.is_available():       torch.cuda.synchronize()
except Exception:
    pass

E_edges  = np.geomspace(E_GeV.min(), E_GeV.max(), E_GeV.size + 1)
CZ_edges = np.linspace(cosz.min(),  cosz.max(),  cosz.size  + 1)

def draw_panel(ax, Z, label_tex, text_color, fontsize=20):
    pc = ax.pcolormesh(
        E_edges, CZ_edges, Z,
        vmin=0.0, vmax=1.0, shading="auto",
        cmap="inferno_r"
    )
    ax.set_xscale("log")

    # add grid lines
    # ax.grid(True, which="both", color="w", alpha=0.25, lw=0.5)
    ax.grid(True, which="both", color="w", alpha=0.3, lw=0.4, ls="--")
    # optional: add minor ticks for better readability
    # ax.minorticks_on()

    ax.text(0.96, 0.96, label_tex,
            transform=ax.transAxes, ha="right", va="top",
            color=text_color, fontsize=fontsize, weight="bold")
    return pc

# --- create figure with constrained layout (better with colorbars) ---
fig, axs = plt.subplots(
    2, 2, figsize=(9.8, 8.0),
    dpi=150,
    constrained_layout=True
)

# draw all panels
m0 = draw_panel(axs[0,0], P_mue,      r"$P_{\nu_\mu \rightarrow \nu_e}$",      "black")
_  = draw_panel(axs[0,1], P_mumu,     r"$P_{\nu_\mu \rightarrow \nu_\mu}$",    "white")
_  = draw_panel(axs[1,0], P_mue_bar,  r"$P_{\bar{\nu}_\mu \rightarrow \bar{\nu}_e}$",  "black")
_  = draw_panel(axs[1,1], P_mumu_bar, r"$P_{\bar{\nu}_\mu \rightarrow \bar{\nu}_\mu}$","white")

# axis labels
for ax in axs[0,:]:
    ax.set_xlabel("")
for ax in axs[:,1]:
    ax.set_ylabel("")
for ax in axs[1,:]:
    ax.set_xlabel(r"$E_\nu$ [GeV]")
for ax in axs[:,0]:
    ax.set_ylabel(r"$\cos\theta_z$")

# single colorbar (automatically aligned to full figure height)
cbar = fig.colorbar(m0, ax=axs, location="right", fraction=0.05, pad=0.03)
cbar.set_label("Oscillation probability", labelpad=10, fontsize=13)

# plt.savefig("../figures/matter_prem_test.pdf") # too heavy
plt.savefig("../figures/matter_prem_test.jpg", dpi=150) if not os.environ.get("CI") else None
plt.show()
