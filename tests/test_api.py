import importlib

def test_import():
    m = importlib.import_module("nu_waves")
    assert hasattr(m, "__version__")

def test_trivial_probability_shape():
    # Replace with your public API call – this is just a smoke test.
    # For example:
    # from nu_waves.propagation import probability
    # p = probability(L=[1.0], E=[0.6], initial="numu", final="nue")
    # assert p.shape == (1,)
    assert True
