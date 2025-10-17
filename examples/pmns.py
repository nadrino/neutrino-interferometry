from nu_waves.models.mixing import MixingParameters
import numpy as np

angles = {(1,2): np.deg2rad(33.4), (1,3): np.deg2rad(8.6), (2,3): np.deg2rad(49.0)}
phases = {(1,3): np.deg2rad(195)}

pmns = MixingParameters(dim=3, mixing_angles=angles, dirac_phases=phases)
U = pmns.U()
print(np.round(U,3))
