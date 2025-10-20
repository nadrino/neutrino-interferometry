# examples/non_adiabatic_supernova.py
import numpy as np
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.matter.supernova import CoreCollapseSN
# If you already integrated the patched function:
# from nu_waves.matter.msw import landau_zener_for_pair

# ---------- PMNS + spectrum ----------
# PDG-ish NH params
theta12 = np.deg2rad(33.44)
theta13 = np.deg2rad(8.57)
dm21 = 7.42e-5     # eV^2
dm31 = 2.517e-3    # eV^2 (NH)
E_MeV = 10.0

angles = {(1, 2): theta12, (1, 3): theta13, (2, 3): 0.0}
pmns = Mixing(dim=3, mixing_angles=angles)
U = pmns.get_mixing_matrix()

spec = Spectrum(n=3, m_lightest=0.0)
spec.set_dm2({(2, 1): dm21, (3, 1): dm31})
m2 = spec.get_m2()

# ---------- SN profile ----------
# Narrow shock → sizable non-adiabatic H resonance
sn = CoreCollapseSN(shock_radius_km=7000.0, shock_width_km=8.0, shock_drop=0.2, Ye=0.5)

# radius grid for plotting eigenvalues
r = np.geomspace(30.0, 1.5e5, 1400)  # km
Ve = sn.Ve(r)

# ---------- build H(r) and diagonalize (for visuals) ----------
E_eV = 1e6 * E_MeV
H_vac = U @ np.diag(m2 / (2.0 * E_eV)) @ np.conjugate(U.T)  # eV
H_list = np.zeros((r.size, 3, 3), dtype=np.complex128)
H_list[:] = H_vac[np.newaxis, :, :]
for k, v in enumerate(Ve):
    H_list[k, 0, 0] += v  # charged-current potential diag(Ve, 0, 0)

evals = np.linalg.eigvalsh(H_list)  # (n,3), sorted ascending

# ---------- Landau–Zener jump probability at H ----------
Pc_H, rH = sn.parke_Pc('H', E_MeV, dm21, dm31, theta12, theta13)
Pc_L, rL = sn.parke_Pc('L', E_MeV, dm21, dm31, theta12, theta13)  # should be ~0 here

# ---------- mass-state fractions along r (NH, ν_e at production) ----------
def smooth_step(x, x0, frac_width=0.02):
    if x0 is None: return np.zeros_like(x)
    w = frac_width * x0
    return 0.5 * (1 + np.tanh((x - x0) / w))

S_H = smooth_step(r, rH)
S_L = smooth_step(r, rL)  # small effect

# Adiabatic: P_H=0 → ν3 outside
p3_ad = 1.0 - 0.0 * S_H
p2_ad = 0.0 * S_H * (1.0 - 0.0) + 0.0 * S_L * (1.0 - S_H)
p1_ad = 1.0 - p2_ad - p3_ad

# Non-adiabatic: P_H = Pc_H
p3_na = 1.0 - Pc_H * S_H
p2_na = Pc_H * S_H * (1.0 - Pc_L) + Pc_L * S_L * (1.0 - S_H)
p1_na = 1.0 - p2_na - p3_na

# ---------- plots ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

def panel(ax, p1, p2, p3, title):
    ax.plot(r, p1, label=r'$\nu_1$')
    ax.plot(r, p2, label=r'$\nu_2$')
    ax.plot(r, p3, label=r'$\nu_3$')
    for rr, txt in [(rH, 'H'), (rL, 'L')]:
        if rr is not None:
            ax.axvline(rr, ls='--', color='k', alpha=0.6)
            ax.text(rr*1.02, 0.05, txt, rotation=90, va='bottom', ha='left')
    ax.set_xscale('log'); ax.set_xlim(r[0], r[-1]); ax.set_ylim(0, 1.0)
    ax.set_xlabel('r [km]'); ax.set_title(title); ax.grid(True, which='both', alpha=0.25)

panel(ax[0], p1_ad, p2_ad, p3_ad, f'Adiabatic (E={E_MeV:.0f} MeV)\n$P_H=0$')
panel(ax[1], p1_na, p2_na, p3_na, f'Non-adiabatic with shock (E={E_MeV:.0f} MeV)\n$P_H={Pc_H:.2f}$')
ax[0].set_ylabel('Mass-state fraction'); ax[1].legend(loc='lower right', frameon=False)
plt.tight_layout(); plt.show()

print(f"H-resonance: Pc_H = {Pc_H:.3f} at r_H ≈ {rH:.0f} km; L-resonance: Pc_L = {Pc_L:.2e}")
