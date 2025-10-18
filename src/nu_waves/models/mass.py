import numpy as np

class Mass:
    """
    Neutrino mass spectrum container.

    Initialize with:
        Spectrum(n=3, m_lightest=0.01)

    Then inject Δm²_ij values via:
        set_dm2({(2,1): 7.4e-5, (3,1): 2.517e-3})

    Attributes
    ----------
    m_lightest : float
        Lightest mass in eV (optional, default 0)
    dm2_matrix : np.ndarray
        Antisymmetric matrix of Δm²_ij = m_i² − m_j²
    """

    def __init__(self, n: int, m_lightest: float = 0.0):
        if n < 2:
            raise ValueError("Number of mass states must be ≥ 2.")
        self.m_lightest = float(m_lightest)
        self.dm2_matrix = np.zeros((n, n), dtype=float)

    @property
    def n(self):
        return self.dm2_matrix.shape[0]

    # ---------- Input of Δm² ----------
    def set_dm2(self, dm2_dict: dict):
        """
        Set Δm²_ij values, checking for consistency and redundancy.
        Example:
            spec.set_dm2({(2,1): 7.4e-5, (3,1): 2.517e-3})
        """

        # --- Loop over entries to validate indices and redundancy ---
        seen_pairs = set()
        for (i, j), val in dm2_dict.items():
            # 1. Invalid or self-referencing indices
            if not (1 <= i <= self.n and 1 <= j <= self.n):
                raise IndexError(f"indices ({i},{j}) out of range for n={self.n}")
            if i == j:
                raise ValueError(f"Invalid Δm²({i},{j}): i and j must differ.")

            # 2. Detect redundant or antisymmetric duplicates
            if (i, j) in seen_pairs or (j, i) in seen_pairs:
                raise ValueError(f"Redundant or conflicting entry for ({i},{j}) / ({j},{i}).")

            seen_pairs.add((i, j))

        # 3. Too many independent entries (redundant information)
        if len(seen_pairs) > self.n - 1:
            raise ValueError(
                f"Too many Δm² entries ({len(seen_pairs)}) for n={self.n}. "
                f"Maximum independent differences is {self.n - 1}."
            )

        # --- passed validation, ready for injection below ---


        """
        Update the internal Δm² matrix with new values, ensuring global transitivity
        and antisymmetry.

        Behavior:
          - Values in dm2_dict override previous ones.
          - Missing entries are inferred from transitivity.
          - Slight inconsistencies are corrected (auto-heal).
        """

        n = self.n
        tol = 1e-10
        M = np.copy(self.dm2_matrix)

        # --- 1. Direct injection of user values ---
        for (i, j), val in dm2_dict.items():
            M[i - 1, j - 1] = val
            M[j - 1, i - 1] = -val

        # --- 2. Build connectivity graph ---
        adjacency = {k: set() for k in range(n)}
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]) and abs(M[i, j]) > 0:
                    adjacency[i].add(j)

        # --- 3. Choose a reference (first connected node) ---
        ref = 0
        visited = {ref}
        queue = [ref]
        delta = np.zeros(n)
        delta[:] = np.nan
        delta[ref] = 0.0

        # --- 4. Propagate Δ_i = Δm²_{i,ref} using BFS ---
        while queue:
            j = queue.pop(0)
            for k in adjacency[j]:
                if np.isnan(M[k, j]):
                    continue
                new_val = delta[j] + M[k, j]
                if np.isnan(delta[k]):
                    delta[k] = new_val
                    visited.add(k)
                    queue.append(k)
                else:
                    # check consistency
                    if abs(delta[k] - new_val) > tol:
                        delta[k] = 0.5 * (delta[k] + new_val)  # smooth correction

        if np.any(np.isnan(delta)):
            raise ValueError("Incomplete Δm² network: some states are disconnected.")

        # --- 5. Fill / correct all entries by transitivity ---
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i, j] = 0.0
                    continue
                expected = delta[i] - delta[j]
                if np.isnan(M[i, j]) or abs(M[i, j] - expected) > tol:
                    M[i, j] = expected
                    M[j, i] = -expected

        # --- 6. Final antisymmetrization & store ---
        self.dm2_matrix = 0.5 * (M - M.T)


    def get_dm2(self, i: int, j: int) -> float:
        """Return Δm²_ij = m_i² − m_j² (1-based indices)."""
        return float(self.dm2_matrix[i-1, j-1])

    def get_m2(self):
        """
        Compute the vector of mass-squared values [m1², m2², ...]
        from the antisymmetric Δm² matrix and the lightest mass.

        The reference state (lightest) is identified automatically
        as the one with the smallest average m² offset.
        """
        M = self.dm2_matrix
        n = M.shape[0]

        # --- identify lightest state (min mean Δm²) ---
        # For a consistent antisymmetric M, the lightest has negative mean shifts
        mean_shift = np.mean(M, axis=1)
        ref = np.argmin(mean_shift)  # index of lightest (0-based)

        # --- reconstruct absolute m² values ---
        m2_lightest = self.m_lightest ** 2
        m2 = m2_lightest + M[:, ref]

        # --- sanity checks ---
        if np.any(m2 < -1e-18):
            raise ValueError(f"Inconsistent Δm² matrix: negative m² values found {m2}")

        return m2

    def get_m(self):
        return np.sqrt(self.get_m2())

    # ---------- Utilities ----------
    def summary(self):
        print(f"Spectrum with {self.n} mass states:")
        print(f"  Lightest mass = {self.m_lightest:.6f} eV")
        print(f"  Masses = {np.round(self.get_m(), 6)} eV")
        print("Δm² matrix [eV²]:")
        print(np.round(self.dm2_matrix, 6))


if __name__ == "__main__":

    print("Providing 2,1 and 3,1:")
    spec = Mass(n=3, m_lightest=0.01)
    spec.set_dm2({
        (2, 1): 7.42e-5,
        (3, 1): 2.517e-3
    })
    spec.summary()
    print("Δm²_21 =", spec.get_dm2(2, 1))
    print("Δm²_31 =", spec.get_dm2(3, 1))
    print("Δm²_32 =", spec.get_dm2(3, 2))

    # coherence tests
    print("Providing 2,1 and 3,2:")
    spec = Mass(n=3, m_lightest=0.01)
    spec.set_dm2({
        (2, 1): 7.42e-5,
        (3, 2): 0.0024428
    })
    spec.summary()
    print("Δm²_21 =", spec.get_dm2(2, 1))
    print("Δm²_31 =", spec.get_dm2(3, 1))
    print("Δm²_32 =", spec.get_dm2(3, 2))

    print("Providing 3,1 and 3,2:")
    spec = Mass(n=3, m_lightest=0.01)
    spec.set_dm2({
        (3, 1): 2.517e-3,
        (3, 2): 0.0024428
    })
    spec.summary()
    print("Δm²_21 =", spec.get_dm2(2, 1))
    print("Δm²_31 =", spec.get_dm2(3, 1))
    print("Δm²_32 =", spec.get_dm2(3, 2))

    print("Inverted ordering:")
    spec.set_dm2({
        (2, 1): 7.42e-5,
        (3, 1): -2.517e-3
    })
    spec.summary()
    print("Δm²_21 =", spec.get_dm2(2, 1))
    print("Δm²_31 =", spec.get_dm2(3, 1))
    print("Δm²_32 =", spec.get_dm2(3, 2))

    print("With sterile:")
    spec = Mass(n=4, m_lightest=0.01)
    spec.set_dm2({
        (3, 1): 2.517e-3,
        (3, 2): 0.0024428,
        (4, 1): 1,
    })
    spec.summary()

