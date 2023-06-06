import numpy as np
from muxsim import MuxSim

NUM_CELLS = 1000
NUM_GUIDES = 100


def test_init():
    ms = MuxSim(
        num_cells=NUM_CELLS,
        num_guides=NUM_GUIDES,
    )
    assert ms.num_cells == NUM_CELLS
    assert ms.num_guides == NUM_GUIDES
    assert ms.umi_sums.size == NUM_CELLS
    assert ms.null_assignments.size == NUM_CELLS
    assert ms.assignments.size == NUM_CELLS
    assert ms.dual_assignments.size == NUM_CELLS
    assert ms.assignments.max() <= NUM_GUIDES
    assert ms.dual_assignments.max() <= NUM_GUIDES
    assert ms.assignments.min() == -1
    assert ms.dual_assignments.min() == -1
    assert ms.null_assignments.max() == 1
    assert ms.null_assignments.min() == 0


def test_masks():
    ms = MuxSim(num_cells=1000, num_guides=100)

    idx_dual = np.flatnonzero(ms.dual_assignments >= 0)
    idx_single = np.flatnonzero(ms.assignments >= 0)
    idx_null = np.flatnonzero(ms.null_assignments == 1)

    assert (
        np.mean(ms.mask_null & ms.mask_assignment) == 0
    ), "No assignments should be in the null set"
    assert (
        np.mean(ms.mask_null & ms.mask_dual) == 0
    ), "No dual assignments should be in the null set"
    assert (
        np.mean(ms.mask_null & ms.mask_single) == 0
    ), "No single assignments should be in the null set"

    assert (
        np.isin(idx_single, idx_null).mean() == 0.0
    ), "All assignments must not be in the null set"
    assert (
        np.isin(idx_dual, idx_single).mean() == 1.0
    ), "All dual assignments must be in the single set"
    assert (
        np.isin(idx_dual, idx_null).mean() == 0.0
    ), "All dual assignments must not be in the null set"


def test_gen():
    ms = MuxSim(num_cells=1000, num_guides=100)
    gen = ms.sample()
    assert gen.shape == (ms.num_cells, ms.num_guides)
