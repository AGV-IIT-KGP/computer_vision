import numpy as np
from agv_cv.edge_detection import edgeDetection
from numpy.testing import assert_equal, assert_almost_equal
from scipy.misc import imread

test = imread('../../images/lena.bmp')
result = imread('../../images/test_edge.png')

def test_edge_detection():
	assert_equal(edgeDetection(test),result)
