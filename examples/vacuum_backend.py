import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
import nu_waves.utils.flavors as flavors

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.get_mixing_matrix()
print(np.round(U_pmns, 3))

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})
spec.summary()

osc = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2())

print("Testing shape of VacuumOscillator.probability")
assert osc.probability(L_km=295, E_GeV=np.linspace(0.2,2,5)).shape == (5,3,3)


print("Testing high energy of VacuumOscillator.probability")
P_hi = osc.probability(L_km=295, E_GeV=1e6)
np.testing.assert_allclose(P_hi, np.eye(3), atol=1e-12)

print("Testing binning of VacuumOscillator.probability")
P = osc.probability(L_km=295, E_GeV=np.linspace(0.2,2,50), flavor_emit=None, flavor_det=None)
np.testing.assert_allclose(P.sum(axis=-2), 1.0, atol=1e-12)


from nu_waves.backends import make_torch_backend
torch_backend = make_torch_backend(seed=123, use_complex64=True)  # MPS if available

# E_min, E_max = 0.2, 3.0
# Enu_list = np.linspace(E_min, E_max, 20000)
# osc = VacuumOscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2(), backend=torch_backend)
# P_mps = osc.probability(
#     L_km=295, E_GeV=Enu_list,
#     alpha=flavors.muon,
#     beta=[flavors.electron, flavors.muon, flavors.tau],
#     antineutrino=False
# )
#
# P_np = torch_backend.from_device(P_mps)

# CPU/NumPy
print("Computing with NumPy")
osc_np = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2())
P_np = osc_np.probability(L_km=295, E_GeV=np.linspace(0.2,2.0,200), flavor_emit=1, flavor_det=1)

# MPS
print("Computing with MPS")
osc_mps = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2(), backend=make_torch_backend(seed=0))
P_mps = osc_mps.probability(L_km=295, E_GeV=np.linspace(0.2,2.0,200), flavor_emit=1, flavor_det=1)

# 1) Check dtype/device of the key tensors on MPS
print("eigvals dtype:", osc_mps.backend.from_device(osc_mps.backend.xp.asarray(0)).dtype)  # sanity

# 2) High-E identity on both backends
E_hi = np.array([1e6])
P_np_hi  = osc_np.probability(L_km=295, E_GeV=E_hi, flavor_emit=None, flavor_det=None)
P_mps_hi = osc_mps.probability(L_km=295, E_GeV=E_hi, flavor_emit=None, flavor_det=None)
# np.testing.assert_allclose(P_np_hi, torch_backend.from_device(P_mps_hi), rtol=1e-4, atol=1e-5)

# 3) Unitarity on MPS alone (catches phase mishandling)
E = np.linspace(0.2, 2.0, 200)
P_mps_full = osc_mps.probability(L_km=295, E_GeV=E, flavor_emit=None, flavor_det=None)
# np.testing.assert_allclose(torch_backend.from_device(P_mps_full).sum(axis=-2), 1.0, atol=2e-6)


# Compute probabilities:
# α = 1 (νμ source), β = [1,2,3] → (νμ, νe, ντ)
E_min, E_max = 0.2, 3.0
# n_points = 20000000 # GPU is much faster
n_points = 200
Enu_list = np.linspace(E_min, E_max, n_points)
P_mps = osc_mps.probability(
    L_km=295, E_GeV=Enu_list,
    flavor_emit=flavors.muon,
    flavor_det=[flavors.electron, flavors.muon, flavors.tau],
    antineutrino=False
)
P = P_mps
# P = osc_np.probability(
#     L_km=295, E_GeV=Enu_list,
#     alpha=flavors.muon,
#     beta=[flavors.electron, flavors.muon, flavors.tau],
#     antineutrino=False
# )

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
disable_draw = False
if not disable_draw:
    plt.figure(figsize=(6.5, 4.0))

    # plt.plot(Enu_list, P[:, flavors.electron], label=r"$P_{\mu e}$ appearance", lw=2)
    # plt.plot(Enu_list, P[:, flavors.muon], label=r"$P_{\mu\mu}$ disappearance", lw=2)
    # plt.plot(Enu_list, P[:, flavors.tau], label=r"$P_{\mu\tau}$ appearance", lw=2)
    plt.plot(Enu_list, P.sum(axis=1), "--", label="Total probability", lw=1.5)

    plt.xlabel(r"$E_\nu$ [GeV]")
    plt.ylabel(r"Probability")
    plt.title(r"MPS!!")
    plt.xlim(E_min, E_max)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()




