import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from numpy.testing import assert_array_equal

from agv_cv.Lane_Detector.Hough import hough_transform
from agv_cv.Lane_Detector.GrassRemoval import *

hough_image = imread("/images/hough_grass.png")
img = imread("/images/lane.png")
def test_hough():
    gray = rgb2gray(img)
    gray = normalize(gray)  
    gray = gaussian_blur(gray)
    final = thresholding(gray,200)
    final = kernel_iteration(gray,180)
    final = hough_transform(final, 5)
    assert_array_equal(final, hough_image, err_msg = "Incorrect output")
    
    
