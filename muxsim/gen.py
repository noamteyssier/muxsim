import numpy as np


class MuxSim:
    def __init__(
        self,
        num_cells: int = 10000,
        num_guides: int = 100,
        n: float = 10.0,
        p: float = 0.1,
        null_rate: float = 0.3,
        dual_rate: float = 0.2,
        random_state: int = 42,
    ):
        self.num_cells = num_cells
        self.num_guides = num_guides
        self.n = n
        self.p = p
        self.null_rate = null_rate
        self.dual_rate = dual_rate
        self.random_state = random_state

        np.random.seed(self.random_state)
        self.umi_sums = self._gen_umi_sums()
        self.null_assignments = self._gen_null_assignments()
        self.assignments = self._gen_assignments(self.null_assignments)
        self.dual_assignments = self._gen_dual_assignments(self.assignments)

        self.mask_null = self.null_assignments == 1
        self.mask_dual = self.dual_assignments != -1
        self.mask_assignment = self.assignments != -1
        self.mask_single = self.mask_assignment & ~self.mask_dual

        self.mean_umi = self.umi_sums.mean()

    def _gen_umi_sums(self) -> np.ndarray:
        """Generate UMI sums for each cell/guide pair."""
        return np.random.negative_binomial(self.n, self.p, size=self.num_cells)

    def _gen_null_assignments(self) -> np.ndarray:
        return np.random.binomial(1, self.null_rate, size=self.num_cells)

    def _gen_assignments(self, null: np.ndarray) -> np.ndarray:
        """Generate assignments for each cell/guide pair."""
        assignment = np.random.choice(
            np.arange(self.num_guides), size=self.num_cells, replace=True
        )
        assignment[null == 1] = -1
        return assignment

    def _gen_dual_assignments(self, assignments: np.ndarray) -> np.ndarray:
        """Generate dual assignments for each cell/guide pair."""
        single_idx = np.flatnonzero(assignments != -1)
        num_single = single_idx.size

        # Only generate dual assignments for cells with single assignments
        mask = np.random.binomial(1, self.dual_rate, size=num_single)
        single_mask = single_idx[mask != 1]

        # Generate dual assignments
        secondary = np.random.choice(
            np.arange(self.num_guides), size=self.num_cells, replace=True
        )

        # Set all null assignments to -1
        secondary[assignments == -1] = -1

        # Set all dual assignments that are not singles to -1
        secondary[single_mask] = -1

        return secondary

    def __repr__(self):
        return f"""MuxSim(
    num_cells={self.num_cells}, 
    num_guides={self.num_guides},
    n={self.n},
    p={self.p},
    null_rate={self.null_rate},
    dual_rate={self.dual_rate},
    random_state={self.random_state}
    mean_umi={self.mean_umi})"""

    def sample(self, signal: float = 10.0, background: float = 0.01):
        """Sample from Multinomial distribution."""
        freq = np.repeat(background, self.num_guides)
        freq /= freq.sum()
        umi = np.zeros((self.num_cells, self.num_guides), dtype=int)
        for i in range(self.num_cells):
            subfreq = freq.copy()
            if self.assignments[i] != -1:
                subfreq[self.assignments[i]] = signal * background
                if self.dual_assignments[i] != -1:
                    subfreq[self.dual_assignments[i]] = signal * background
                subfreq /= subfreq.sum()
                print(subfreq.sum())
            umi[i] = np.random.multinomial(self.umi_sums[i], subfreq)
        return umi
