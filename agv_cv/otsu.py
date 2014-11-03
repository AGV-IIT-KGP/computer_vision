import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def otsu_thresh(grey):
    [rows, cols] = grey.shape

    histo, bin_edges = np.histogram(grey, bins = range(255))
    histo = np.array(histo)
    hist_image = histo*1.0/(rows*cols)
    ut = 0
    for k in range(1, 255):
        ut = ut + k*hist_image[k-1]
    w = 0
    u = 0
    values = np.zeros(255)

    for k in range(1, 255):
        w = w + hist_image[k-1]
        u = u + k*hist_image[k-1]
        values[k] = ((ut*w - u)**2)/(w*(1-w))

    otsu = np.argmax(values)
    return otsu
