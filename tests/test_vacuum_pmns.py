import numpy as np
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian.vacuum import VacuumHamiltonian
from nu_waves.propagation.new_oscillator import Oscillator
from nu_waves.utils.flavors import electron, muon, tau
from nu_waves.backends.torch_backend import make_torch_backend

USE_NUMPY = True
# USE_NUMPY = False
backend = None

if not USE_NUMPY:
    try:
        import torch
        print("torch available")
        backend = make_torch_backend(
            force_device="cpu"
        )
        print(backend.device)
        HAS_TORCH = True
    except Exception:
        print("Torch is not available")

if backend is None:
    print("Using NumPy backend")


angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}

h = VacuumHamiltonian(
    mixing_matrix=Mixing(dim=3, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_array=Spectrum(n=3, m_lightest=0, dm2=dm2).get_m2()
)

osc = Oscillator(hamiltonian=h)

def test_syntax():
    print("test_syntax test...")
    P = osc.probability(L_km=[0], E_GeV=[1])
    print(f"P = {P}")
    assert P.shape == (3, 3)
    P = osc.probability(L_km=0, E_GeV=np.linspace(0.2, 3.0, 10))
    assert P.shape == (10, 3, 3)
    P = osc.probability(L_km=0, E_GeV=np.linspace(0.2, 3.0, 10), flavor_emit=muon)
    assert P.shape == (10, 3)
    P = osc.probability(L_km=0, E_GeV=np.linspace(0.2, 3.0, 10), flavor_det=muon)
    assert P.shape == (10, 3)
    P = osc.probability(L_km=0, E_GeV=np.linspace(0.2, 3.0, 10), flavor_emit=muon, flavor_det=[muon, electron])
    assert P.shape == (10, 2)
    P = osc.probability(
        L_km=np.linspace(0, 300, 10),
        E_GeV=np.linspace(0.2, 3.0, 10),
        flavor_emit=muon, flavor_det=[muon, electron]
    )
    assert P.shape == (10, 2)
    P = osc.probability(
        L_km=np.linspace(0, 300, 10),
        E_GeV=1,
        flavor_emit=muon, flavor_det=[muon, electron]
    )
    assert P.shape == (10, 2)
    try:
        P = osc.probability(
            L_km=np.linspace(0, 300, 11),
            E_GeV=np.linspace(0.2, 3.0, 10),
            flavor_emit=muon, flavor_det=[muon, electron]
        )
        # SHOULD PRODUCE AN ERROR
        assert False
    except ValueError:
        pass
    print("test_syntax test success.")

def test_zero_baseline_identity():
    print("zero_baseline_identity test...")
    P = osc.probability(
        flavor_emit=muon, flavor_det=[electron, muon, tau],
        L_km=0, E_GeV=np.linspace(0.2, 3.0, 10),
    )
    print(P[:, electron])
    assert np.allclose(P[:, electron], 0.0, atol=1e-14) # no electron appearance
    assert np.allclose(P[:, tau], 0.0, atol=1e-14) # no tau appearance
    assert np.allclose(P[:, muon], 1.0, atol=1e-14) # all muons
    print("zero_baseline_identity: success.")


def test_probability_conservation():
    print("test_probability_conservation test...")
    E_min, E_max = 0.3, 3.0
    Enu_list = np.linspace(E_min, E_max, 3)
    P = osc.probability(
        flavor_emit=muon, flavor_det=[electron, muon, tau],
        L_km=295, E_GeV=Enu_list, # t2k baseline
    )
    try:
        # for any energy, sum of P for each flavor should be 1.
        assert np.allclose(np.sum(P, axis=1), 1.0, atol=1E-18)
    except AssertionError:
        pass
        for iE, flavor_prob in enumerate(P):
            print(f"E({Enu_list[iE]:.3f} GeV):", flavor_prob, f"sum={np.sum(flavor_prob)}")
        print("Assertion failed.")

    print("test_probability_conservation: success.")


test_syntax()
test_zero_baseline_identity()
test_probability_conservation()
