import numpy as np
from agv_cv.pointoperators import inversion
from numpy.testing import assert_equal, assert_almost_equal

image = np.array([[0, 0, 1, 3, 5],
                  [0, 1, 4, 3, 4],
                  [1, 2, 5, 4, 1],
                  [2, 4, 5, 2, 1],
                  [4, 5, 1, 0, 0]], dtype=int)


def test_inversion():
    assert_equal(inversion(image), -image)
