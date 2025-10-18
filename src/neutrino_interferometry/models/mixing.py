from dataclasses import dataclass, field
import numpy as np

@dataclass
class MixingParameters:
    """
    Generic N-flavor neutrino mixing matrix definition.
    Angles (theta) and phases (delta, majorana) must be given explicitly.
    """
    dim: int = 3
    mixing_angles: dict = field(default_factory=dict)
    dirac_phases: dict = field(default_factory=dict)
    majorana_phases: list = field(default_factory=list)

    def U(self, include_majorana=False):
        """Return the full complex mixing matrix U (n×n)."""
        U = np.eye(self.dim, dtype=np.complex128)
        for (i,j) in sorted(self.mixing_angles.keys()):
            mixing_angle, dirac_phase = self.mixing_angles[(i, j)], self.dirac_phases.get((i, j), 0.0)
            s, c = np.sin(mixing_angle), np.cos(mixing_angle)
            R = np.eye(self.dim, dtype=np.complex128)
            R[i-1,i-1] = c; R[j-1,j-1] = c
            R[i-1,j-1] = s*np.exp(-1j*dirac_phase); R[j-1,i-1] = -s*np.exp(1j*dirac_phase)
            U = U @ R
        if include_majorana and any(self.majorana_phases):
            M = np.diag([np.exp(1j*a/2) for a in [0] + self.majorana_phases])
            U = U @ M
        return U


# inline example
if __name__ == "__main__":
    print("Running mixing.py test")

    angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49.0)}
    phases = {(1, 3): np.deg2rad(195)}

    pmns = MixingParameters(dim=3, mixing_angles=angles, dirac_phases=phases)
    U = pmns.U()
    print(np.round(U, 3))

    # example with a sterile state
    angles[(1, 4)] = np.arcsin(np.sqrt(0.1))
    # angles[(2,4)] = 0
    # angles[(3,4)] = 0

    # two additional dirac phases
    # phases[(1,4)] = np.deg2rad(90) # \delta_{14}
    # phases[(2,4)] = np.deg2rad(0) # \delta_{24}

    pmns_3p1 = MixingParameters(dim=4, mixing_angles=angles, dirac_phases=phases)
    U = pmns_3p1.U()
    print("U (3+1):")
    print(np.round(U, 3))


