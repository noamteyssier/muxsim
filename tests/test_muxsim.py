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
    assert ms.moi.size == NUM_CELLS
    assert len(ms.assignments) == NUM_CELLS


def test_gen():
    ms = MuxSim(num_cells=1000, num_guides=100)
    gen = ms.sample()
    assert gen.shape == (ms.num_cells, ms.num_guides)
