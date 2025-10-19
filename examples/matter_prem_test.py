import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import VacuumOscillator
from nu_waves.matter.prem import PREMModel
from nu_waves.matter.profile import MatterProfile
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

# sanity test for layer ordering
prof = MatterProfile.from_segments(
    rho_gcm3=[2.8, 11.0], Ye=[0.5, 0.467], lengths_km=[3000.0, 2000.0]
)
osc.set_layered_profile(prof)
E = np.linspace(2,8,200)
P_fwd = osc.probability(L_km=sum(l.weight for l in prof.layers), E_GeV=E, alpha=1, beta=0)

# reverse the profile
prof_rev = MatterProfile.from_segments(
    rho_gcm3=[11.0, 2.8], Ye=[0.467, 0.5], lengths_km=[2000.0, 3000.0]
)
osc.set_layered_profile(prof_rev)
P_rev = osc.probability(L_km=sum(l.weight for l in prof_rev.layers), E_GeV=E, alpha=1, beta=0)

if torch_backend:
    P_rev = torch_backend.from_device(P_rev)
    P_fwd = torch_backend.from_device(P_fwd)

print("max |ΔP| (fwd vs rev) =", np.max(np.abs(P_fwd - P_rev)))  # should be >> 1e-3

L = 8000.0
E = np.linspace(1, 10, 400)
P_NO = osc.probability(L_km=L, E_GeV=E, alpha=1, beta=0)              # neutrinos, NO
P_IO = osc.probability(L_km=L, E_GeV=E, alpha=1, beta=0, antineutrino=True)  # ν̄ in IO has the resonance

if torch_backend:
    P_NO = torch_backend.from_device(P_NO)
    P_IO = torch_backend.from_device(P_IO)
print("P_mu->e max (NO, ν):", P_NO.max(), "  P_mu->e max (IO, ν̄):", P_IO.max())

E_GeV = np.logspace(-1, 2, 400)     # x
cosz  = np.linspace(-1.0, 1.0, 240)     # y (upgoing)
prem  = PREMModel()

# test for thickness
prof = prem.profile_from_coszen(+0.3, h_atm_km=15.0)
print("L_tot(downgoing) =", sum(L.weight for L in prof.layers))  # ≈ 15 km

for cz in (-1e-3, +1e-3):
    prof = prem.profile_from_coszen(cz, h_atm_km=15.0)
    print(cz, "L_atm =",
          sum(L.weight for L in prof.layers if L.rho_gcm3==0.0))

prof1 = prem.profile_from_coszen(-1, scheme="prem_layers")
prof2 = prem.profile_from_coszen(-1, scheme="hist_density", n_bins=4000, nbins_density=60)
osc.set_layered_profile(prof1); P1 = osc.probability(L_km=sum(l.weight for l in prof1.layers), E_GeV=E, alpha=1, beta=0)
osc.set_layered_profile(prof2); P2 = osc.probability(L_km=sum(l.weight for l in prof2.layers), E_GeV=E, alpha=1, beta=0)

if torch_backend:
    P2 = torch_backend.from_device(P2)
    P1 = torch_backend.from_device(P1)

print("max |ΔP| prem vs hist =", np.max(np.abs(P1-P2)))




# P_mue  = np.zeros((cosz.size, E_GeV.size))
# P_mumu = np.zeros_like(P_mue)

# choose one:
SCHEME = "prem_layers"      # exact PREM shells
# SCHEME = "hist_density"   # fine histogram of density along path

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
    P_mue_i      = osc.probability(L_km=L_tot, E_GeV=E_GeV, alpha=1, beta=0, antineutrino=False)
    P_mumu_i     = osc.probability(L_km=L_tot, E_GeV=E_GeV, alpha=1, beta=1, antineutrino=False)
    P_mue_bar_i  = osc.probability(L_km=L_tot, E_GeV=E_GeV, alpha=1, beta=0, antineutrino=True)
    P_mumu_bar_i = osc.probability(L_km=L_tot, E_GeV=E_GeV, alpha=1, beta=1, antineutrino=True)

    # 4) move from device if needed
    if torch_backend is not None:
        P_mue[iy]      = torch_backend.from_device(P_mue_i)
        P_mumu[iy]     = torch_backend.from_device(P_mumu_i)
        P_mue_bar[iy]  = torch_backend.from_device(P_mue_bar_i)
        P_mumu_bar[iy] = torch_backend.from_device(P_mumu_bar_i)
    else:
        P_mue[iy], P_mumu[iy] = P_mue_i, P_mumu_i
        P_mue_bar[iy], P_mumu_bar[iy] = P_mue_bar_i, P_mumu_bar_i

# (optional) synchronize GPU timing before plotting
try:
    import torch
    if torch.backends.mps.is_available(): torch.mps.synchronize()
    elif torch.cuda.is_available():       torch.cuda.synchronize()
except Exception:
    pass

# ---------- plotting (2×2) ----------
def draw(ax, X, Y, Z, title, vmax=1.0):
    E_edges = np.geomspace(X.min(), X.max(), X.size + 1)
    CZ_edges = np.linspace(Y.min(), Y.max(), Y.size + 1)
    pc = ax.pcolormesh(E_edges, CZ_edges, Z, vmin=0.0, vmax=vmax, shading="auto", cmap="inferno")
    ax.set_xscale("log")
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(r"$\cos\theta_z$")
    ax.set_title(title)
    cbar = plt.colorbar(pc, ax=ax, pad=0.01)
    cbar.set_label("Probability")

fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=150, constrained_layout=True)
draw(axs[0,0], E_GeV, cosz, P_mue,      r"$P(\nu_\mu\to\nu_e)$")
draw(axs[0,1], E_GeV, cosz, P_mumu,     r"$P(\nu_\mu\to\nu_\mu)$")
draw(axs[1,0], E_GeV, cosz, P_mue_bar,  r"$P(\bar{\nu}_\mu\to\bar{\nu}_e)$")
draw(axs[1,1], E_GeV, cosz, P_mumu_bar, r"$P(\bar{\nu}_\mu\to\bar{\nu}_\mu)$")
plt.show()
