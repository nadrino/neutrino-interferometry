from dataclasses import dataclass
from enum import Enum

from tomlkit.items import Bool


class Basis(Enum):
    MASS = "mass"               # natural base for neutrino propagation
    FLAVOR = "flavor"           # natural base for interaction
    MATTER = "matter_custom"    # instantaneous eigen state in a given matter environment


@dataclass
class WaveFunction:
    current_basis: Basis
    values: any                 # complex xp.ndarray (NumPy or Torch). Shape: (nE, nF)
    eigen_vectors: any = None   # optional: eigenvector matrix defining this basis (nF, nF)

    def __post_init__(self):
        # placeholder
        pass

    @property
    def shape(self):
        return self.values.shape

    @property
    def n_flavors(self):
        return self.shape[-1]

    def copy(self, xp):
        return WaveFunction(
            current_basis=self.current_basis,
            values=xp.copy(self.values),
            eigen_vectors=None if self.eigen_vectors is None else xp.copy(self.eigen_vectors),
        )

    def to_basis(self, target_basis: Basis, eigen_vectors_dagger):
        """
        Rotate the wavefunction to a new basis defined by `eigen_vectors_dagger`.

        Parameters
        ----------
        target_basis : Basis
            Target basis label (for bookkeeping)
        eigen_vectors_dagger : xp.ndarray
            Columns are eigen_vectors defining the *target* basis.
            Shape (nF, nF).
        """
        # H' = U† H U  → ψ' = U† ψ
        # shapes: ( (nF, nF) @ (nE, nFd, nF, 1) ) -> (nE, nFd, nF, 1) -> (nE, nFd, nF)
        self.values = (eigen_vectors_dagger @ self.values[..., None])[..., 0]
        self.current_basis = target_basis
