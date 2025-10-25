from nu_waves.hamiltonian.matter_constant import MatterLayer, MatterProfile
from dataclasses import dataclass
import numpy as np


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
        if h_km <= 0: return 0.0
        R, Rp = self.R_earth_km, self.R_earth_km + h_km
        b = self._impact_parameter_km(cosz)
        if b >= Rp:  # ray misses the atmosphere
            return 0.0
        seg = lambda r: np.sqrt(max(r * r - b * b, 0.0))
        # ONE segment from production altitude Rp down to the surface R, irrespective of sign(cosz)
        return seg(Rp) - seg(R)

    # --- layer builders ---
    def profile_from_coszen(self, cosz: float,
                            scheme: str = "prem_layers",
                            n_bins: int = 800,
                            nbins_density: int = 24,
                            merge_tol: float = 0.0,
                            h_atm_km: float = 15.0) -> MatterProfile:
        cz = float(cosz)
        Ltot = self._chord_length_km(cz)
        b = self._impact_parameter_km(cz)

        layers: list[MatterLayer] = []

        # --- Earth chord (only if upgoing) ---
        if Ltot > 0.0:
            half = 0.5 * Ltot

            def r_of_t(t):
                return np.sqrt(b * b + t * t)

            if scheme == "prem_layers":
                # cut at PREM boundaries intersected by the chord
                t_pts = [-half, +half]
                for rb in self.prem_boundaries_km[1:-1]:
                    if rb > b:
                        dt = np.sqrt(rb * rb - b * b)
                        t_pts += [-dt, +dt]
                t_pts = np.array(sorted(tp for tp in t_pts if -half <= tp <= half))
                for t0, t1 in zip(t_pts[:-1], t_pts[1:]):
                    dL = float(t1 - t0)
                    r_mid = r_of_t(0.5 * (t0 + t1))
                    rho_mid = float(self.rho(r_mid))
                    Ye_mid = float(self.Ye(r_mid))
                    layers.append(MatterLayer(rho_mid, Ye_mid, dL))

            elif scheme == "hist_density":
                t_edges = np.linspace(-half, +half, n_bins + 1)
                t_mid = 0.5 * (t_edges[:-1] + t_edges[1:])
                dL = np.diff(t_edges)
                r_mid = r_of_t(t_mid)
                rho_mid = self.rho(r_mid)
                Ye_mid = self.Ye(r_mid)

                rmin, rmax = rho_mid.min(), rho_mid.max()
                edges = np.linspace(rmin, rmax, nbins_density + 1)
                idx = np.minimum(np.searchsorted(edges, rho_mid, side="right") - 1, nbins_density - 1)

                accL = accR = accY = 0.0
                prev = idx[0]

                def flush():
                    nonlocal accL, accR, accY
                    if accL > 0.0:
                        layers.append(MatterLayer(accR / accL, accY / accL, accL))
                        accL = accR = accY = 0.0

                for i in range(len(dL)):
                    if idx[i] != prev:
                        flush()
                        prev = idx[i]
                    accL += float(dL[i])
                    accR += float(dL[i] * rho_mid[i])
                    accY += float(dL[i] * Ye_mid[i])
                flush()

                if merge_tol > 0 and len(layers) > 1:
                    merged = [layers[0]]
                    for nxt in layers[1:]:
                        cur = merged[-1]
                        if abs(nxt.rho_in_g_per_cm3 - cur.rho_in_g_per_cm3) <= merge_tol:
                            Lsum = cur.weight + nxt.weight
                            merged[-1] = MatterLayer(
                                (cur.rho_in_g_per_cm3 * cur.weight + nxt.rho_in_g_per_cm3 * nxt.weight) / Lsum,
                                (cur.Ye * cur.weight + nxt.Ye * nxt.weight) / Lsum,
                                Lsum
                            )
                        else:
                            merged.append(nxt)
                    layers = merged
            else:
                raise ValueError(f"Unknown scheme '{scheme}'")
        # else: no Earth layers for cz>=0

        # --- Atmosphere around the Earth chord (both signs of cosz) ---
        L_atm = self._atm_path_km(cosz, h_atm_km)
        if L_atm > 0.0:
            atm = MatterLayer(0.0, self.Ye_mantle, L_atm)
            if Ltot > 0.0:  # upgoing: atmosphere is on the FAR side, before Earth
                layers = [atm] + layers
            else:  # downgoing: atmosphere is on the NEAR side, straight to detector
                layers = [atm]

        # Ensure non-empty profile to keep the engine happy
        if not layers:
            layers = [MatterLayer(0.0, self.Ye_mantle, 0.0)]

        return MatterProfile(layers=layers, slicing="absolute")
