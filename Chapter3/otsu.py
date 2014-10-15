import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

def greyscale(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def otsu_thresh(grey):
	[rows, cols] = grey.shape

	histo, bin_edges = np.histogram(grey, bins = range(256))
	histo = np.array(histo)
	hist_image = histo*1.0/(rows*cols)
	ut = 0
	for k in range(254):
		ut = ut + k*hist_image[k]
	w = 0
	u = 0
	values = np.zeros(256)

	for k in range(254):
		w = w + hist_image[k]
		u = u + k*hist_image[k]		
		values[k] = ((ut*w - u)**2)/(w*(1-w))
	otsu = np.argmax(values)
	return otsu

if __name__ == "__main__":
	
	img = imread('lena.png')
	plt.imshow(img)
	plt.show()

	grey = greyscale(img)
	plt.imshow(grey, cmap = cm.gray)
	plt.show()

	otsu = otsu_thresh(grey)
	print otsu

	new_img = np.empty([grey.shape[0], grey.shape[1]])

	new_img[grey>=otsu] = 255
	new_img[grey<otsu] = 0

	plt.imshow(new_img, cmap = cm.gray)
	plt.show()



