import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

def rgb2grey(pixel):
	return math.floor(0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2])

def greyscale(img):
	greyimg = np.zeros((img.shape[0], img.shape[1])) 
	for rownum in range(len(img)):
		for colnum in range(len(img[rownum])):
			greyimg[rownum][colnum] = rgb2grey(img[rownum][colnum])
	return greyimg

def otsu_thresh(grey):
	[rows, cols] = grey.shape
	
	histo = np.zeros(256)
	for y in range(rows):
		for x in range(cols):
			histo[grey[y][x]] = histo[grey[y][x]] + 1

	hist_image = histo*1.0/(rows*cols)
	ut = 0
	for k in range(256):
		ut = ut + k*hist_image[k]
	w = 0
	u = 0
	values = np.zeros(256)

	for k in range(255):
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



