import numpy as np
from agv_cv import *
from numpy.testing import assert_equal, assert_almost_equal
from scipy.misc import imread
from scipy import ndimage

lena=imread('images/lenag.bmp')

def test_load_image():
	lena = imread('images/lena.bmp')
	assert lena != None

def test_inversion():
    assert_equal(inversion(lena), -lena)

def average():
	lena_avg=mean(lena,5)
	assert_equal(lena_avg,ndimage.uniform_filter(lena, size=11))

def gaussian():
	lena_gau=gauss_mean(lena,3,1)
	assert_equal(lena_gau,ndimage.gaussian_filter(lena,sigma=1))

def med():
	lena_med=medianSQR(lena,3)
	assert_equal(lena_med,ndimage.median_filter(lena, 3))

