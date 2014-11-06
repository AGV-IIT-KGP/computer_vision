import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imread
import scipy.signal as sg


def rgb2grey(image):
	rows,cols,channels = image.shape
	if (channels == 3):
		image = np.mean(image,axis=2)
	return image

def edgeDetection(image):
    image = rgb2grey(image)
    rows,cols = image.shape
    result = np.zeros ((rows,cols))
    template = np.array([[2,-1],[-1,0]])
    result = sg.convolve(image,template)
    return result


if __name__=="__main__":
    lena = imread('../images/lena.bmp')
    lena1 = edgeDetection(lena)
    plt.imshow(lena1,cmap=cm.Greys_r)
    plt.show()
