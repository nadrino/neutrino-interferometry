import numpy as np


def _argmin_abs(x):
    i = int(np.nanargmin(np.abs(x)))
    return i


def landau_zener_for_pair(r_km: np.ndarray,
                          evals: np.ndarray,   # shape (n, N), eigenvalues λ_k(r)
                          i: int, j: int,
                          min_separation=1e-12):
    """
    Generic Landau–Zener estimate for an avoided crossing between eigenmodes i and j.

    Parameters
    ----------
    r_km : (n,)  monotonic path radii
    evals: (n,N) instantaneous eigenvalues λ_k(r) from your hermitian H(r) [same units]
    i,j  : mode indices (0-based)
    min_separation : small floor to avoid division by zero in slope

    Returns
    -------
    dict with:
      has_cross : bool
      idx_star  : int  index nearest to crossing
      r_star    : float  crossing radius (interpolated)
      gap       : float  minimal |λ_i - λ_j|   (same units as evals)
      slope     : float  d/dr (λ_i - λ_j) at r_star
      Pc        : float  exp(-π gap^2 / (2 |slope|))
    """
    r = np.asarray(r_km, float)
    lam = np.asarray(evals, float)
    n, N = lam.shape
    if r.ndim != 1 or r.size != n or n < 3:
        raise ValueError("Shapes must satisfy: r_km -> (n,), evals -> (n,N) with n>=3")

    d = lam[:, i] - lam[:, j]           # δλ_ij(r)
    # locate minimum of |δλ|
    k0 = _argmin_abs(d)
    # reject if the minimum is at the boundary (can’t estimate slope)
    if k0 == 0 or k0 == n-1:
        return dict(has_cross=False, idx_star=None, r_star=np.nan,
                    gap=np.nan, slope=np.nan, Pc=0.0)

    # Quadratic interpolation for r_star and gap
    r0, r1, r2 = r[k0-1], r[k0], r[k0+1]
    d0, d1, d2 = d[k0-1], d[k0], d[k0+1]
    # Fit a parabola a x^2 + b x + c to (r,d) near k0
    A = np.array([[r0*r0, r0, 1.0],
                  [r1*r1, r1, 1.0],
                  [r2*r2, r2, 1.0]], dtype=float)
    y = np.array([d0, d1, d2], dtype=float)
    try:
        a, b, c = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        # fallback: use central values
        r_star = r1
        gap = abs(d1)
        slope = (d2 - d0) / max(r2 - r0, min_separation)
    else:
        # vertex of parabola
        if a != 0.0:
            r_star = -b/(2*a)
            # clamp to [r0,r2]
            r_star = float(np.clip(r_star, min(r0,r2), max(r0,r2)))
        else:
            r_star = r1
        gap = abs(a*r_star*r_star + b*r_star + c)
        slope = abs(2*a*r_star + b)

    # ... after we found k0 and (r0,r1,r2),(d0,d1,d2) and r_star
    # gap at r_star (use parabolic value if fit succeeded, else |d1|)
    try:
        a, b, c = np.linalg.solve(A, y)
        gap = abs(a * r_star * r_star + b * r_star + c)
    except np.linalg.LinAlgError:
        gap = abs(d1)

    # robust central-difference slope at the minimum neighborhood
    dr = max(r2 - r0, 1e-18)
    slope = (d2 - d0) / dr
    slope = float(np.sign(slope) * max(abs(slope), 1e-18))  # clamp

    # if gap or slope non-finite, bail out
    if not np.isfinite(gap) or not np.isfinite(slope):
        return dict(has_cross=False, idx_star=None, r_star=np.nan,
                    gap=np.nan, slope=np.nan, Pc=0.0)

    Pc = float(np.exp(-np.pi * gap * gap / (2.0 * abs(slope))))
    Pc = float(np.clip(Pc, 0.0, 1.0))
    return dict(has_cross=True, idx_star=k0, r_star=r_star, gap=gap, slope=slope, Pc=Pc)

