import numpy as np

def landau_zener_for_pair(r_km: np.ndarray,
                          evals: np.ndarray,   # shape (n, N), eigenvalues λ_k(r)
                          i: int, j: int,
                          min_separation: float = 1e-12,
                          window: int = 7):
    """
    Landau–Zener jump probability for an avoided crossing between modes i and j,
    computed *only* from the adiabatic eigenvalues λ_k(r).

    Key idea: near the crossing, the adiabatic gap
      g_ad(r) = |λ_j(r) - λ_i(r)|
    behaves like  sqrt( v^2 (r-r*)^2 + g_min^2 ),
    so g_ad(r)^2 is a parabola with curvature  v^2 .
    Fitting a quadratic to g_ad^2 around the minimum gives v = |dΔ/dr|_r*,
    the derivative of the *diabatic* detuning needed in the LZ exponent.

    Returns a dict with:
      has_cross : bool
      idx_star  : index near the crossing
      r_star    : float, location of the crossing (interpolated)
      gap       : float, minimal adiabatic gap g_ad(r*)  [same units as evals]
      slope     : float, v = |dΔ/dr| at r*               [units of evals per km]
      Pc        : float, Landau–Zener jump probability  Pc = exp(-π * gap^2 / (2 v))
    """
    r = np.asarray(r_km, dtype=float)
    lam = np.asarray(evals, dtype=float)
    if lam.ndim != 2 or r.ndim != 1 or lam.shape[0] != r.size:
        raise ValueError("Shapes must satisfy: r_km -> (n,), evals -> (n, N)")

    n = r.size
    if n < 5:
        # need a few points to fit a curvature
        return dict(has_cross=False, idx_star=None, r_star=np.nan,
                    gap=np.nan, slope=np.nan, Pc=0.0)

    # Adiabatic gap between the two modes
    d = lam[:, j] - lam[:, i]
    g = np.abs(d)

    # Locate index of minimal adiabatic gap
    k0 = int(np.argmin(g))
    if k0 < 2 or k0 > n - 3:
        # too close to edge to estimate curvature robustly
        return dict(has_cross=False, idx_star=k0, r_star=r[k0],
                    gap=float(g[k0]), slope=np.nan, Pc=0.0)

    # Build a symmetric window around k0 (odd number of points)
    window = max(5, window | 1)  # ensure odd and >=5
    half = min(window // 2, k0, n - 1 - k0)
    sl = slice(k0 - half, k0 + half + 1)

    x = r[sl] - r[k0]                # center x around the minimum
    y = g[sl] ** 2                   # square of the adiabatic gap

    # Quadratic fit: y ≈ A x^2 + B x + C
    try:
        A, B, C = np.polyfit(x, y, 2)
    except np.linalg.LinAlgError:
        # fallback: coarse curvature & central values
        dr = r[k0 + 1] - r[k0 - 1]
        if dr == 0:
            return dict(has_cross=False, idx_star=k0, r_star=r[k0],
                        gap=float(g[k0]), slope=np.nan, Pc=0.0)
        # second derivative of y ~ (y_{+} - 2 y_0 + y_{-}) / (Δr)^2  = 2 A
        ypp = (y[half + 1] - 2 * y[half] + y[half - 1]) / (dr ** 2)
        A = 0.5 * ypp
        B = 0.0
        C = y[half]

    if not np.isfinite(A) or A <= 0:
        # no proper avoided crossing curvature detected
        return dict(has_cross=False, idx_star=k0, r_star=r[k0],
                    gap=float(g[k0]), slope=np.nan, Pc=0.0)

    # Vertex of the parabola gives r_star and minimal gap
    x_star = -B / (2.0 * A)
    # keep it inside the window for stability
    x_star = float(np.clip(x_star, x[0], x[-1]))
    r_star = float(r[k0] + x_star)

    gap_min_sq = float(A * x_star * x_star + B * x_star + C)
    gap_min_sq = max(gap_min_sq, 0.0)
    gap_min = float(np.sqrt(gap_min_sq))

    # v = |dΔ/dr| at the crossing from curvature of g^2: v = sqrt(A)
    v = float(np.sqrt(max(A, 0.0)))
    v = max(v, float(min_separation))  # floor to avoid div-by-zero

    # Landau–Zener jump probability; here gap_min is the adiabatic minimum (≈ 2|coupling|)
    Pc = float(np.exp(-np.pi * gap_min * gap_min / (2.0 * v)))
    Pc = float(np.clip(Pc, 0.0, 1.0))

    return dict(has_cross=True, idx_star=k0, r_star=r_star,
                gap=gap_min, slope=v, Pc=Pc)
