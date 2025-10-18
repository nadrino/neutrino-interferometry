import numpy as np

KM_TO_EVINV = 5.067730716e9  # eV^-1 per km (1 km / ħc in natural units)

def propagator_vacuum(H: np.ndarray, L: float) -> np.ndarray:
    """
    S = exp(-i H L) via diagonalisation: H = V Λ V^† → S = V e^{-iΛL} V^†
    H: (N,N) ou (nE,N,N). Retourne S de même forme.
    """
    if H.ndim == 2:
        w, V = np.linalg.eigh(H)
        phase = np.exp(-1j * w * L * KM_TO_EVINV)
        return V @ (phase[:, None] * V.conj().T)
    # batch sur E
    nE, N, _ = H.shape
    S = np.empty_like(H, dtype=np.complex128)
    for k in range(nE):
        w, V = np.linalg.eigh(H[k])
        phase = np.exp(-1j * w * L * KM_TO_EVINV)
        S[k] = V @ (phase[:, None] * V.conj().T)
    return S


def probability_alpha_to_beta(S: np.ndarray, alpha: int, beta: int) -> np.ndarray:
    """
    P_{α→β} = |S_{β,α}|^2. α,β en 1-based. Supporte batch E.
    """
    a, b = alpha-1, beta-1
    if S.ndim == 2:
        return np.abs(S[b, a])**2
    return np.abs(S[:, b, a])**2