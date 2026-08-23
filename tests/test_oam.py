import numpy as np

from experiments.chapter_4.ensemble_size_convergence import (
    create_input_beam,
)

from configs.chapter_2 import (
    L_WINDOW,
    N_GRID,
)


def test_create_input_beam():
    grid, field = create_input_beam()

    assert grid.x.shape == (N_GRID,)
    assert grid.y.shape == (N_GRID,)

    assert field.shape == (
        N_GRID,
        N_GRID,
    )

    assert np.all(
        np.isfinite(field)
    )

    assert np.iscomplexobj(field)

    assert np.max(
        np.abs(field)
    ) > 0.0

    dx = L_WINDOW / N_GRID

    assert np.isclose(
        grid.dx,
        dx,
    )
