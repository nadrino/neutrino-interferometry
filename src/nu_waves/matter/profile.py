from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Sequence, Literal


@dataclass(frozen=True)
class MatterLayer:
    rho_gcm3: float
    Ye: float
    weight: float        # either absolute length [km] or fraction [0..1]
    kind: Literal["absolute", "fraction"] = "fraction"

@dataclass
class MatterProfile:
    layers: list[MatterLayer]

    @staticmethod
    def from_fractions(rho_gcm3: Sequence[float], Ye: Sequence[float], fractions: Sequence[float]) -> "MatterProfile":
        fr = np.asarray(fractions, float)
        if not np.isclose(fr.sum(), 1.0, atol=1e-12):
            raise ValueError("fractions must sum to 1")
        Ls = [MatterLayer(r, y, w, "fraction") for r, y, w in zip(rho_gcm3, Ye, fr)]
        return MatterProfile(Ls)

    @staticmethod
    def from_segments(rho_gcm3: Sequence[float], Ye: Sequence[float], lengths_km: Sequence[float]) -> "MatterProfile":
        Ls = [MatterLayer(r, y, L, "absolute") for r, y, L in zip(rho_gcm3, Ye, lengths_km)]
        return MatterProfile(Ls)

    def resolve_dL(self, L_total_km_array) -> list[np.ndarray]:
        """
        Map total baselines (array-like) to per-layer ΔL_k arrays.
        Returns list of arrays, one per layer, each shaped like L_total_km_array.
        - fraction layers scale with L_total
        - absolute layers ignore L_total (assumed consistent with physics setup)
        """
        Ltot = np.asarray(L_total_km_array, float)
        dLs = []
        for layer in self.layers:
            if layer.kind == "fraction":
                dLs.append(layer.weight * Ltot)
            else:
                dLs.append(np.full_like(Ltot, layer.weight, dtype=float))
        return dLs
