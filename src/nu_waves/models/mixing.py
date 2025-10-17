from dataclasses import dataclass, field
import numpy as np

@dataclass
class MixingParameters:
    """
    Generic N-flavor neutrino mixing matrix definition.
    Angles (theta) and phases (delta, majorana) must be given explicitly.
    """
    n_neutrinos: int = 3
    mixing_angles: dict = field(default_factory=dict)
    dirac_phases: dict = field(default_factory=dict)
    majorana_phases: list = field(default_factory=list)

    def U(self, include_majorana=False):
        """Return the full complex mixing matrix U (n×n)."""
        U = np.eye(self.n_neutrinos, dtype=np.complex128)
        for (i,j) in sorted(self.mixing_angles.keys()):
            mixing_angle, dirac_phase = self.mixing_angles[(i, j)], self.dirac_phases.get((i, j), 0.0)
            s, c = np.sin(mixing_angle), np.cos(mixing_angle)
            R = np.eye(self.n_neutrinos, dtype=np.complex128)
            R[i-1,i-1] = c; R[j-1,j-1] = c
            R[i-1,j-1] = s*np.exp(-1j*dirac_phase); R[j-1,i-1] = -s*np.exp(1j*dirac_phase)
            U = U @ R
        if include_majorana and any(self.majorana_phases):
            M = np.diag([np.exp(1j*a/2) for a in [0] + self.majorana_phases])
            U = U @ M
        return U
