import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import VacuumOscillator
from nu_waves.matter.profile import MatterProfile
import nu_waves.utils.flavors as flavors

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(dim=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.get_mixing_matrix()

# Masses, normal ordering
spec = Spectrum(n=3, m_lightest=0.)
spec.set_dm2({(2, 1): 7.42e-5, (3, 2): 0.0024428})

# oscillator
osc = VacuumOscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2())

osc.use_vacuum()
P_vac = osc.probability(L_km=295, E_GeV=np.linspace(0.3,3,200), alpha=1, beta=0)
osc.set_layered_profile(MatterProfile.from_fractions([0.0],[0.5],[1.0]))
P_chk = osc.probability(L_km=295, E_GeV=np.linspace(0.3,3,200), alpha=1, beta=0)
print("Checking null density")
np.testing.assert_allclose(P_vac, P_chk, atol=1e-12)

prof_AB = MatterProfile.from_fractions([2.8, 11.0], [0.5, 0.467], [0.7, 0.3])
osc.set_layered_profile(prof_AB)
P_full = osc.probability(L_km=1000, E_GeV=np.linspace(0.3,5,120), alpha=None, beta=None)
print("Checking Unitarity")
np.testing.assert_allclose(P_full.sum(axis=-2), 1.0, atol=2e-6)

prof_BA = MatterProfile.from_fractions([11.0, 2.8], [0.467, 0.5], [0.3, 0.7])
osc.set_layered_profile(prof_AB); P_AB = osc.probability(L_km=1000, E_GeV=np.linspace(0.3,5,200), alpha=1, beta=0)
osc.set_layered_profile(prof_BA); P_BA = osc.probability(L_km=1000, E_GeV=np.linspace(0.3,5,200), alpha=1, beta=0)
print("Checking ordering of layers -> should have an effect")
print(np.max(np.abs(P_AB - P_BA)))
assert np.max(np.abs(P_AB - P_BA)) > 1e-3


