from neutrino_interferometry.models.mixing import MixingParameters
import numpy as np

angles = {(1,2): np.deg2rad(33.4), (1,3): np.deg2rad(8.6), (2,3): np.deg2rad(49.0)}
phases = {(1,3): np.deg2rad(195)}

pmns = MixingParameters(dim=3, mixing_angles=angles, dirac_phases=phases)
U = pmns.U()
print(np.round(U,3))


# example with a sterile state
angles[(1,4)] = np.arcsin(np.sqrt(0.1))
# angles[(2,4)] = 0
# angles[(3,4)] = 0

# two additional dirac phases
# phases[(1,4)] = np.deg2rad(90) # \delta_{14}
# phases[(2,4)] = np.deg2rad(0) # \delta_{24}

pmns_3p1 = MixingParameters(dim=4, mixing_angles=angles, dirac_phases=phases)
U = pmns_3p1.U()
print("U (3+1):")
print(np.round(U,3))

