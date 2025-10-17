from dataclasses import dataclass, field
import numpy as np

@dataclass
class MixingParameters:
    """
    General N-flavor neutrino mixing model (Dirac or Majorana).

    Attributes
    ----------
    n_flavors : int
        Number of mass eigenstates.
    theta : dict[(i,j), float]
        Mixing angles θ_ij (radians, i<j).
    delta_cp : dict[(i,j), float]
        Dirac CP phases δ_ij (radians, optional).
    dm2 : dict[(i,j), float]
        Mass-squared differences Δm²_ij = m_i² − m_j² [eV²].
    majorana : list[float]
        Optional Majorana phases α_k (radians), applied as diag(1, e^{iα21/2}, e^{iα31/2}, ...).
    hierarchy : str
        "normal" or "inverted".
    """

    n_flavors: int = 3
    theta: dict = field(default_factory=dict)
    delta_cp: dict = field(default_factory=dict)
    dm2: dict = field(default_factory=dict)
    majorana: list = field(default_factory=list)
    hierarchy: str = "normal"

    def __post_init__(self):
        # Default to PDG 3-flavor Dirac parameters
        if not self.theta and self.n_flavors == 3:
            self.theta = {(1,2): np.deg2rad(33.44),
                          (1,3): np.deg2rad(8.57),
                          (2,3): np.deg2rad(49.2)}
            self.delta_cp = {(1, 3): np.deg2rad(195)}
            self.dm2 = {(2,1): 7.42e-5,
                        (3,1): 2.517e-3 if self.hierarchy == "normal" else -2.498e-3}

        # If fewer Majorana phases than required, pad with zeros
        if len(self.majorana) < self.n_flavors - 1:
            self.majorana += [0.0] * (self.n_flavors - 1 - len(self.majorana))

    # ------------------------------------------------------------

    def pmns_matrix(self, include_majorana=True):
        """Return the full mixing matrix U (unitary, N×N)."""
        n = self.n_flavors
        U = np.eye(n, dtype=np.complex128)

        for (i,j) in sorted(self.theta.keys()):
            theta_array = self.theta[(i,j)]
            delta_cp = self.delta_cp.get((i, j), 0.0)
            s, c = np.sin(theta_array), np.cos(theta_array)

            R = np.eye(n, dtype=np.complex128)
            R[i-1,i-1] = c
            R[j-1,j-1] = c
            R[i-1,j-1] = s * np.exp(-1j * delta_cp)
            R[j-1,i-1] = -s * np.exp(1j * delta_cp)
            U = U @ R

        if include_majorana and any(self.majorana):
            M = np.eye(n, dtype=np.complex128)
            for k in range(1, n):
                M[k, k] = np.exp(1j * self.majorana[k-1] / 2)
            U = U @ M

        return U

    # ------------------------------------------------------------

    def mass_spectrum(self):
        """Return mass-squared eigenvalues (up to additive constant)."""
        m2 = np.zeros(self.n_flavors)
        for (i,j), val in self.dm2.items():
            m2[i-1] = m2[j-1] + val
        return m2

    # ------------------------------------------------------------

    def summary(self, include_majorana=True):
        print(f"Neutrino model: {self.n_flavors} flavors ({self.hierarchy})")
        for (i,j), theta_array in self.theta.items():
            delta_cp = self.delta_cp.get((i, j), 0.0)
            print(f"θ{i}{j} = {np.rad2deg(theta_array):.3f}°, δ{i}{j} = {np.rad2deg(delta_cp):.3f}°")
        for (i,j), val in self.dm2.items():
            print(f"Δm²_{i}{j} = {val:.3e} eV²")
        if include_majorana and any(self.majorana):
            for k, α in enumerate(self.majorana, start=2):
                print(f"α{k}1 = {np.rad2deg(α):.3f}°")
