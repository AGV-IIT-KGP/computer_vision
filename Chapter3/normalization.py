import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

def greyscale(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def normalization(grey):
	#creating a greyscale version
	[rows, cols] = grey.shape

	grey_max = np.amax(grey)
	grey_min = np.amin(grey)
	width  = grey_max - grey_min

	#normalizing the greyscale image
	normal = np.zeros((grey.shape[0], grey.shape[1]))
	normal = np.floor((grey-grey_min)*255*1.0/width)

	return normal
	
if __name__ == "__main__":
	
	img = imread('lena.png')
	plt.imshow(img)
	plt.show()

	grey = greyscale(img)
	plt.imshow(grey, cmap = cm.gray)
	plt.show()
	#histogram of grayscale image
	plt.hist(grey)
	plt.show()

	normal = normalization(grey)

	plt.imshow(normal, cmap = cm.gray)
	plt.show()
	#histogram of normalized image
	plt.hist(normal)
	plt.show()


