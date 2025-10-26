from nu_waves.globals.backend import Backend
from dataclasses import dataclass, field


class Mixing:
    """
    Generic N-flavor neutrino mixing matrix definition.
    Angles (theta) and phases (delta, majorana) must be given explicitly.
    """
    def __init__(
          self,
          n_neutrinos: int,
          mixing_angles: dict = None,
          dirac_phases: dict = None,
          majorana_phases: dict = None
    ):
        self.n_neutrinos = n_neutrinos
        self.mixing_angles = mixing_angles if mixing_angles is not None else dict()
        self.dirac_phases = dirac_phases if dirac_phases is not None else dict()
        self.majorana_phases = majorana_phases if majorana_phases is not None else dict()

    def build_mixing_matrix(self, include_majorana: bool = False):
        """
        Return the full complex mixing matrix U (dim x dim).

        PDG convention is enforced on the active 3x3 sub-block:
            U_active = R23 * U13(delta) * R12
        for any dim >= 3. All remaining rotations (e.g. with sterile states)
        are then applied in the user-provided insertion order.
        """
        U = Backend.xp().eye(self.n_neutrinos, dtype=Backend.complex_dtype())

        # Build the ordered list of rotation pairs to apply (right-multiplication)
        provided = list(self.mixing_angles.keys())  # preserves insertion order (Py3.7+)
        angles_ordered: list[tuple[int, int]] = []

        if self.n_neutrinos >= 3:
            pdg_triple = [(2, 3), (1, 3), (1, 2)]
            # First, apply PDG order for those pairs that are actually provided
            angles_ordered.extend([p for p in pdg_triple if p in self.mixing_angles])

        # Then append all the remaining pairs as provided (without sorting)
        angles_ordered.extend([p for p in provided if p not in angles_ordered])

        # Apply rotations in that order (right-multiply: rotations act on mass columns)
        for (i, j) in angles_ordered:
            theta = Backend.xp().asarray(self.mixing_angles[(i, j)], dtype=Backend.real_dtype())
            delta = Backend.xp().asarray(self.dirac_phases.get((i, j), 0.0), dtype=Backend.real_dtype())
            s, c = Backend.xp().sin(theta), Backend.xp().cos(theta)

            R = Backend.xp().eye(self.n_neutrinos, dtype=Backend.complex_dtype())
            ii, jj = i - 1, j - 1
            R[ii, ii] = c
            R[jj, jj] = c
            # PDG sign convention (R23 has -s in (3,2); implemented generically here)
            R[ii, jj] = s * Backend.xp().exp(-1j * delta)
            R[jj, ii] = -s * Backend.xp().exp(+1j * delta)

            U = U @ R

        if include_majorana and any(self.majorana_phases):
            # Majorana phases: dim-1 physical phases (first one conventionally 0)
            phases = [0.0] + list(self.majorana_phases)
            phases = Backend.xp().array(phases[:self.n_neutrinos], dtype=float)  # trim/pad
            M = Backend.xp().diag(Backend.xp().exp(0.5j * phases))
            U = U @ M

        return U


# inline example
if __name__ == "__main__":
    import numpy as np
    print("Running mixing.py test")

    angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49.0)}
    phases = {(1, 3): np.deg2rad(195)}

    pmns = Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases)
    U = pmns.build_mixing_matrix()
    print(np.round(U, 3))

    # example with a sterile state
    angles[(1, 4)] = np.arcsin(np.sqrt(0.1))
    # angles[(2,4)] = 0
    # angles[(3,4)] = 0

    # two additional dirac phases
    # phases[(1,4)] = np.deg2rad(90) # \delta_{14}
    # phases[(2,4)] = np.deg2rad(0) # \delta_{24}

    pmns_3p1 = Mixing(n_neutrinos=4, mixing_angles=angles, dirac_phases=phases)
    U = pmns_3p1.build_mixing_matrix()
    print("U (3+1):")
    print(np.round(U, 3))


