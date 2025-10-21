import numpy as np
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator

backend = None

angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}

osc = Oscillator(
    mixing_matrix=Mixing(dim=3, mixing_angles=angles, dirac_phases=phases).get_mixing_matrix(),
    m2_list=Spectrum(n=3, dm2=dm2).get_m2(),
    backend=backend,
)


