import numpy as np

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.matter.profile import MatterProfile

# Torch backend is optional
try:
        TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.get_mixing_matrix()

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})

def run_once(backend_name="numpy"):
    if backend_name == "numpy":
        backend = make_numpy_backend(seed=0)
    else:
        backend = make_torch_backend(seed=0, use_complex64=True)

    # Build oscillator
    osc = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2(), backend=backend)

    # Two-layer toy profile (mantle 70%, core 30%) — just to exercise layered propagation
    rho = [2.8, 11.0]    # g/cm^3
    Ye  = [0.50, 0.467]
    frac = [0.7, 0.3]
    prof = MatterProfile.from_fractions(rho, Ye, frac)
    osc.set_layered_profile(prof)

    # Config (DUNE-ish energies, baseline chosen so layers matter)
    L_km = 1300.0
    E = np.linspace(0.3, 5.0, 240)

    # Compute full matrix and selected channels
    P_full = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=None, flavor_det=None)
    P_mue = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=0)
    P_mumu = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=1)

    # Unitarity check: sum over beta equals 1
    if backend_name == "numpy":
        np.testing.assert_allclose(P_full.sum(axis=-2), 1.0, atol=2e-6)
    else:
        # bring to host for the assertion
        np.testing.assert_allclose(P_full.sum(axis=-2), 1.0, atol=2e-5)

    return backend, P_mue, P_mumu

if __name__ == "__main__":
    print("Computing with NumPy backend ...")
    np_backend, P_mue_np, P_mumu_np = run_once("numpy")

    if not TORCH_AVAILABLE:
        print("Torch backend not available; skipping MPS test.")
        raise SystemExit(0)

    torch_backend = make_torch_backend(seed=0, use_complex64=True)
    # Report device
    dev = "mps" if "mps" in str(torch_backend.xp.device) else str(torch_backend.xp.device)
    print(f"Computing with Torch backend on device: {dev}")

    # Reuse the same function but force torch backend inside
    osc_mps = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2(),
                         backend=torch_backend)

    # same profile
    from nu_waves.matter.profile import MatterProfile
    prof = MatterProfile.from_fractions([2.8, 11.0], [0.50, 0.467], [0.7, 0.3])
    osc_mps.set_layered_profile(prof)

    L_km = 1300.0
    E = np.linspace(0.3, 5.0, 240)
    P_mue_mps  = osc_mps.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=0)
    P_mumu_mps = osc_mps.probability(L_km=L_km, E_GeV=E, flavor_emit=1, flavor_det=1)

    np.testing.assert_allclose(P_mue_np,  P_mue_mps,  rtol=5e-4, atol=5e-5)
    np.testing.assert_allclose(P_mumu_np, P_mumu_mps, rtol=5e-4, atol=5e-5)

    print("MPS layered propagation parity: OK")
