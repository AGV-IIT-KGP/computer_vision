import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

def greyscale(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
	
def equalization(grey):
	#creating a greyscale version
	[rows, cols] = grey.shape

	grey_max = np.amax(grey)
	grey_min = np.amin(grey)
	width  = grey_max - grey_min
	
	brightness = np.zeros(256)
	
	for y in range(rows):
		for x in range(cols):
			brightness[grey[y][x]] = brightness[grey[y][x]] + 1
	
	sum = 0
	hist = np.zeros(256)
	for level in range(256):
		sum = sum + brightness[level]
		hist[level] = np.floor((width*sum)/(rows*cols))

	equal = np.zeros((grey.shape[0], grey.shape[1]))
	for r in range(rows):
		for c in range(cols):
			equal[r][c] = hist[grey[r][c]]
	return equal

if __name__ == "__main__":
	
	img = imread('lena.png')
	plt.imshow(img)
	plt.show()

	grey = greyscale(img)
	plt.imshow(grey, cmap = cm.gray)
	plt.show()

	plt.hist(grey)
	plt.show()

	equal = equalization(grey)

	plt.imshow(equal, cmap = cm.gray)
	plt.show()

	plt.hist(equal)
	plt.show()