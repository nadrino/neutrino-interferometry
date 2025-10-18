import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import VacuumOscillator
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

# calculate without matter effects
osc.use_vacuum()
P_vac = osc.probability(L_km=295, E_GeV=np.linspace(0.2,2,50), alpha=1, beta=0)

# compare with density = 0
osc.set_constant_density(rho_gcm3=0.0, Ye=0.5)
P_zero = osc.probability(L_km=295, E_GeV=np.linspace(0.2,2,50), alpha=1, beta=0)

# should be equal
np.testing.assert_allclose(P_vac, P_zero, atol=1e-12)





