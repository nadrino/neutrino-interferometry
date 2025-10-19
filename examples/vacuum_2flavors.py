import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import VacuumOscillator
from nu_waves.backends import make_torch_mps_backend
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style

# toggle GPU
torch_backend = None
# torch_backend = make_torch_mps_backend(seed=0, use_complex64=True)

# sterile test
osc_amplitude = 0.2 # sin^2(2\theta)
angles = {(1, 2): np.arcsin(np.sqrt(osc_amplitude))/2}
pmns = Mixing(dim=2, mixing_angles=angles)
U_pmns = pmns.get_mixing_matrix()
print(np.round(U_pmns, 3))


# 1 eV^2
spec = Spectrum(n=2, m_lightest=0.)
spec.set_dm2({(2, 1): 1})
spec.summary()
m2_diag = np.diag(spec.get_m2())


osc = VacuumOscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2(), backend=torch_backend)

E_fixed = 3E-3
L_min, L_max = 1e-3, 20e-3
L_list = np.linspace(L_min, L_max, 200)
print(L_list)
P = osc.probability(
    L_km=L_list, E_GeV=E_fixed,
    alpha=flavors.electron,
    beta=flavors.electron, # muon could be sterile
    antineutrino=True
)
# print(P)

xp = osc.backend.xp

# Example: fractional Gaussian energy resolution and fixed baseline blur
rng = np.random.default_rng(123)
def baseline_sampler(L_center, n):
    L_center = xp.asarray(L_center)
    print("L_center",L_center)
    # broadcast low/high to center's shape, then append sample axis
    low = (L_center - 0.001)[..., None]
    high = (L_center + 0.001)[..., None]
    return rng.uniform(low=low, high=high, size=L_center.shape + (n,))

def baseline_sampler_gauss(L_center, n):
    """
    Absolute Gaussian smearing with sigma = 0.001 km.
    Works for scalar, vector, or grid L_center; returns shape L_center.shape + (n,).
    """
    Lc = xp.asarray(L_center, dtype=float)
    return rng.normal(loc=Lc[..., None], scale=0.001, size=Lc.shape + (n,))

def energy_sampler_sqrt(E_center, n, a=0.008):
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
    print("out.shape",out.shape)
    return out

# osc.baseline_sampler = baseline_sampler_gauss
osc.energy_sampler = energy_sampler_sqrt
osc.n_samples = 1000
P_damp = osc.probability(
    L_km=L_list, E_GeV=E_fixed,
    alpha=flavors.electron,
    beta=flavors.electron, # muon could be sterile
    antineutrino=True
)



# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.figure(figsize=(6.5, 4.0), dpi=150)

plt.plot(L_list*1000, P, label=r"$P_{e e}$ disappearance", lw=2)
plt.plot(L_list*1000, P_damp, label=r"$P_{e e}$ disappearance (with E smearing)", lw=2)
plt.plot(L_list*1000, [1]*len(L_list), "--", label="Total probability", lw=1.5)

plt.xlabel(r"$L_\nu$ [m]")
plt.ylabel(r"Probability")
plt.title(f"eV$^2$ sterile with $E_\\nu$ = {E_fixed*1000} MeV")
# plt.xlim(L_min, L_max)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig("../figures/vacuum_2flavors.jpg", dpi=150)
plt.show()
