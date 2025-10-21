import numpy as np
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.utils.flavors import electron, muon, tau
from nu_waves.backends.torch_backend import make_torch_backend

backend = None
try:
    import torch
    print("torch available")
    backend = make_torch_backend()
    print(backend.device)
    HAS_TORCH = True
except Exception:
    print("Torch is not available")


angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}

osc = Oscillator(
    mixing_matrix=Mixing(dim=3, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_list=Spectrum(n=3, dm2=dm2).get_m2(),
    backend=backend,
)


def test_zero_baseline_identity():
    print("zero_baseline_identity test...")
    E_min, E_max = 0.2, 3.0
    Enu_list = np.linspace(E_min, E_max, 10)
    P = osc.probability(
        flavor_emit=muon, flavor_det=[electron, muon, tau],
        L_km=0, E_GeV=Enu_list,
    )
    assert np.allclose(P[:, electron], 0.0, atol=0) # no electron appearance
    assert np.allclose(P[:, tau], 0.0, atol=0) # no tau appearance
    assert np.allclose(P[:, muon], 1.0, atol=0) # all muons
    print("zero_baseline_identity: success.")


def test_probability_conservation():
    print("test_probability_conservation test...")
    E_min, E_max = 0.2, 3.0
    Enu_list = np.linspace(E_min, E_max, 10)
    P = osc.probability(
        flavor_emit=muon, flavor_det=[electron, muon, tau],
        L_km=295, E_GeV=Enu_list, # t2k baseline
    )
    # for any energy, sum of P for each flavor should be 1.
    assert np.allclose(np.sum(P, axis=1), 0.0, atol=1E-12)
    print("test_probability_conservation: success.")


test_zero_baseline_identity()
test_probability_conservation()
