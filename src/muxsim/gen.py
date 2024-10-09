import numpy as np


class MuxSim:
    def __init__(
        self,
        num_cells: int = 10000,
        num_guides: int = 100,
        n: float = 10.0,
        p: float = 0.1,
        λ: float = 0.8,
        random_state: int = 42,
    ):
        self.num_cells = num_cells
        self.num_guides = num_guides
        self.n = n
        self.p = p
        self.λ = λ
        self.random_state = random_state

        np.random.seed(self.random_state)
        self.umi_sums = self._gen_umi_sums()
        self.moi = self._gen_moi()
        self.assignments = self._gen_assignments()
        self.mean_umi = self.umi_sums.mean()

    def _gen_umi_sums(self) -> np.ndarray:
        """Generate UMI sums for each cell/guide pair."""
        return np.random.negative_binomial(self.n, self.p, size=self.num_cells)

    def _gen_moi(self) -> np.ndarray:
        """Generate MOI for each cell/guide pair."""
        return np.random.poisson(self.λ, size=self.num_cells)

    def _gen_assignments(self) -> list:
        """Generate assignments for each cell/guide pair."""
        assignment = [
            np.random.choice(self.num_guides, self.moi[i])
            for i in range(self.num_cells)
        ]
        return assignment

    def __repr__(self):
        return f"""MuxSim(
    num_cells={self.num_cells}, 
    num_guides={self.num_guides},
    n={self.n},
    p={self.p},
    λ={self.λ},
    random_state={self.random_state}
    mean_umi={self.mean_umi})"""

    def sample(self, signal: float = 10.0):
        """Sample from Multinomial distribution."""
        background = 1.0 / self.num_guides
        freq = np.repeat(background, self.num_guides)
        umi = np.zeros((self.num_cells, self.num_guides), dtype=int)
        for i in range(self.num_cells):
            subfreq = freq.copy()
            if self.moi[i] != 0:
                subfreq[self.assignments[i]] = signal * background
                subfreq /= subfreq.sum()
            umi[i] = np.random.multinomial(self.umi_sums[i], subfreq)
        return umi
