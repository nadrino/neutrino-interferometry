from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .profile import MatterLayer, MatterProfile

@dataclass
class PREMModel:
    """
    Minimal PREM density model (Dziewonski & Anderson 1981) with two discretization modes:
      - 'prem_layers': cut the chord at PREM radial boundaries and use the local density.
      - 'hist_density': sample along the chord, bin by density (nbins), and merge contiguous bins.

    Densities are in g/cm^3. Electron fraction Ye defaults: mantle≈0.495, core≈0.467.
    """
    R_earth_km: float = 6371.0
    # PREM region boundaries in km (increasing radii)
    # 0–1221.5 (inner core), 1221.5–3480 (outer core), 3480–... mantle shells, crustal layers
    prem_boundaries_km: tuple = (0.0, 1221.5, 3480.0, 5701.0, 5771.0, 5971.0,
                                 6151.0, 6346.6, 6356.0, 6368.0, 6371.0)
    # Ye defaults
    Ye_mantle: float = 0.495
    Ye_core:   float = 0.467

    # --- PREM density polynomials ρ(r) in g/cm^3; x = r/R_earth ---
    def rho(self, r_km: np.ndarray) -> np.ndarray:
        r = np.asarray(r_km, float)
        x = r / self.R_earth_km
        rho = np.empty_like(x)

        # regions defined by r
        b = self.prem_boundaries_km

        # 0–1221.5 km (inner core)
        m = (r >= b[0]) & (r < b[1])
        rho[m] = 13.0885 - 8.8381 * x[m]**2

        # 1221.5–3480 km (outer core)
        m = (r >= b[1]) & (r < b[2])
        rho[m] = 12.5815 - 1.2638 * x[m] - 3.6426 * x[m]**2 - 5.5281 * x[m]**3

        # 3480–5701 km (lower mantle)
        m = (r >= b[2]) & (r < b[3])
        rho[m] = 7.9565 - 6.4761 * x[m] + 5.5283 * x[m]**2 - 3.0807 * x[m]**3

        # 5701–5771 km (TZ 1)
        m = (r >= b[3]) & (r < b[4])
        rho[m] = 5.3197 - 1.4836 * x[m]

        # 5771–5971 km (TZ 2)
        m = (r >= b[4]) & (r < b[5])
        rho[m] = 11.2494 - 8.0298 * x[m]

        # 5971–6151 km (upper mantle low)
        m = (r >= b[5]) & (r < b[6])
        rho[m] = 7.1089 - 3.8045 * x[m]

        # 6151–6346.6 km (upper mantle high)
        m = (r >= b[6]) & (r < b[7])
        rho[m] = 2.6910 + 0.6924 * x[m]

        # 6346.6–6356 km (crust 1)
        m = (r >= b[7]) & (r < b[8])
        rho[m] = 2.900

        # 6356–6368 km (crust 2)
        m = (r >= b[8]) & (r < b[9])
        rho[m] = 2.600

        # 6368–6371 km (ocean)
        m = (r >= b[9]) & (r <= b[10])
        rho[m] = 1.020

        return rho

    def Ye(self, r_km: np.ndarray) -> np.ndarray:
        """Simple two-zone Ye: core vs mantle/crust."""
        r = np.asarray(r_km, float)
        return np.where(r <= 3480.0, self.Ye_core, self.Ye_mantle)

    # --- chord geometry helpers ---
    def _chord_length_km(self, cosz: float) -> float:
        cz = float(cosz)
        return 0.0 if cz >= 0.0 else -2.0 * self.R_earth_km * cz

    def _impact_parameter_km(self, cosz: float) -> float:
        cz = float(cosz)
        return self.R_earth_km * np.sqrt(max(0.0, 1.0 - cz*cz))

    # atmosphere thickness
    def _atm_path_km(self, cosz: float, h_km: float) -> float:
        """Line-of-sight path length through a thin shell from radius R to R+h."""
        if h_km <= 0: return 0.0
        R, Rp = self.R_earth_km, self.R_earth_km + h_km
        b = self._impact_parameter_km(cosz)
        if b >= Rp:  # grazing above atmosphere
            return 0.0

        def seg(r):
            return np.sqrt(max(r * r - b * b, 0.0))

        if cosz >= 0.0:
            # one segment (production→detector)
            return seg(Rp) - seg(R)
        else:
            # two segments (far + near side)
            return 2.0 * (seg(Rp) - seg(R))

    # --- layer builders ---
    def profile_from_coszen(self, cosz: float,
                            scheme: str = "prem_layers",
                            n_bins: int = 800,
                            nbins_density: int = 24,
                            merge_tol: float = 0.0,
                            h_atm_km: float = 15.0) -> MatterProfile:
        """
        Build a MatterProfile for a chord at cosine-zenith 'cosz'.
        scheme:
          - "prem_layers": cut at PREM boundaries intersected by the chord.
          - "hist_density": sample uniformly along the chord (n_bins),
                            bin density into 'nbins_density' bins, and merge contiguous bins.

        merge_tol: optional tolerance on density (g/cm^3) to merge adjacent segments in 'hist_density'.
        """
        Ltot = self._chord_length_km(cosz)
        if Ltot <= 0.0:
            # No Earth crossing: return zero-length layer (vacuum propagation elsewhere if desired)
            return MatterProfile([MatterLayer(self.rho(6371.0), self.Ye(6371.0), 0.0, "absolute")])

        b = self._impact_parameter_km(cosz)
        half = 0.5 * Ltot

        # Parameterize chord by t ∈ [-L/2, +L/2], r(t) = sqrt(b^2 + t^2)
        def r_of_t(t): return np.sqrt(b*b + t*t)

        if scheme == "prem_layers":
            # find intersection t's where r(t) equals a PREM boundary
            t_points = [-half, +half]
            for rb in self.prem_boundaries_km[1:-1]:  # skip center and surface (already in)
                if rb > b:  # intersects the chord if boundary radius > impact parameter
                    dt = np.sqrt(rb*rb - b*b)
                    t_points.append(-dt)
                    t_points.append(+dt)
            t_points = np.array(sorted(tp for tp in t_points if -half <= tp <= half))

            layers: list[MatterLayer] = []
            for t0, t1 in zip(t_points[:-1], t_points[1:]):
                dL = float(t1 - t0)
                r_mid = r_of_t(0.5*(t0 + t1))
                rho_mid = float(self.rho(r_mid))
                Ye_mid  = float(self.Ye(r_mid))
                layers.append(MatterLayer(rho_mid, Ye_mid, dL, "absolute"))

            L_atm = self._atm_path_km(cosz, h_atm_km)
            if L_atm > 0.0:
                if cosz >= 0.0:
                    layers = [MatterLayer(0.0, self.Ye_mantle, L_atm, "absolute")] + layers
                else:
                    half = 0.5 * L_atm
                    layers = [MatterLayer(0.0, self.Ye_mantle, half, "absolute")] + layers + \
                             [MatterLayer(0.0, self.Ye_mantle, half, "absolute")]

            return MatterProfile(layers)

        elif scheme == "hist_density":
            # uniform bins in t, mid-sample density
            t_edges = np.linspace(-half, +half, n_bins+1)
            t_mid   = 0.5 * (t_edges[:-1] + t_edges[1:])
            dL      = np.diff(t_edges)
            r_mid   = r_of_t(t_mid)
            rho_mid = self.rho(r_mid)
            Ye_mid  = self.Ye(r_mid)

            # build density bins from the path's min/max
            rmin, rmax = rho_mid.min(), rho_mid.max()
            edges = np.linspace(rmin, rmax, nbins_density+1)
            # create bin indices; ensure last bin inclusive
            idx = np.minimum(np.searchsorted(edges, rho_mid, side="right")-1, nbins_density-1)

            # merge contiguous bins with same idx, optionally by tolerance
            layers: list[MatterLayer] = []
            acc_L, acc_rho_sum, acc_Ye_sum, acc_n = 0.0, 0.0, 0.0, 0
            prev_idx = idx[0]

            def flush():
                if acc_n == 0: return
                # use average rho/Ye of the segment; alternatively use bin mid: (edges[prev_idx]+edges[prev_idx+1])/2
                rho_avg = acc_rho_sum / acc_n
                Ye_avg  = acc_Ye_sum  / acc_n
                layers.append(MatterLayer(float(rho_avg), float(Ye_avg), float(acc_L), "absolute"))

            for i in range(n_bins):
                same_bin = (idx[i] == prev_idx)
                if not same_bin:
                    flush()
                    acc_L, acc_rho_sum, acc_Ye_sum, acc_n = 0.0, 0.0, 0.0, 0
                    prev_idx = idx[i]
                # merge step
                acc_L        += float(dL[i])
                acc_rho_sum  += float(rho_mid[i])
                acc_Ye_sum   += float(Ye_mid[i])
                acc_n        += 1

            flush()
            # optional additional merge by absolute tolerance on ρ between adjacent layers
            if merge_tol > 0 and len(layers) > 1:
                merged: list[MatterLayer] = []
                cur = layers[0]
                for nxt in layers[1:]:
                    if abs(nxt.rho_gcm3 - cur.rho_gcm3) <= merge_tol:
                        # merge in place
                        totL = cur.weight + nxt.weight
                        rho_avg = (cur.rho_gcm3*cur.weight + nxt.rho_gcm3*nxt.weight) / totL
                        Ye_avg  = (cur.Ye*cur.weight       + nxt.Ye*nxt.weight)         / totL
                        cur = MatterLayer(rho_avg, Ye_avg, totL, "absolute")
                    else:
                        merged.append(cur)
                        cur = nxt
                merged.append(cur)
                layers = merged

            L_atm = self._atm_path_km(cosz, h_atm_km)
            if L_atm > 0.0:
                if cosz >= 0.0:
                    layers = [MatterLayer(0.0, self.Ye_mantle, L_atm, "absolute")] + layers
                else:
                    half = 0.5 * L_atm
                    layers = [MatterLayer(0.0, self.Ye_mantle, half, "absolute")] + layers + \
                             [MatterLayer(0.0, self.Ye_mantle, half, "absolute")]

            return MatterProfile(layers)

        else:
            raise ValueError(f"Unknown scheme '{scheme}'. Use 'prem_layers' or 'hist_density'.")
