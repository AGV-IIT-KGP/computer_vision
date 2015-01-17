import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def greyscale(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])	

def temp_conv(img, temp):
	(irows, icols) = img.shape
	(trows, tcols) = temp.shape
	newimg = np.zeros(img.shape)
	for i in range(trows/2, irows - trows/2):
		for j in range(tcols/2, icols - tcols/2):
			newimg[i][j] = np.sum(img[i-trows/2:i+trows/2+1, j-tcols/2:j+tcols/2+1] * temp)/np.sum(temp)
	np.clip(newimg, 0, 255, newimg)
	return newimg

def average(img, trows, tcols):
	(irows, icols) = img.shape
	newimg = np.zeros(img.shape)
	for i in range(trows/2, irows - trows/2):
		for j in range(tcols/2, icols - tcols/2):
			newimg[i][j] = np.sum(img[i-trows/2:i+trows/2+1, j-tcols/2:j+tcols/2+1] * 1/trows*tcols)
	np.clip(newimg, 0, 255, newimg)
	return newimg

def gaussian_template(trows, tcols, variance):
	temp = np.zeros((trows, tcols))
	ceny = trows/2
	cenx = tcols/2
	sum = 0
	for i in range(trows):
		for j in range(tcols):
			temp[i][j] = math.exp(-( (i-ceny)*(i-ceny) + (j-cenx)*(j-cenx) )/(2*variance*variance))
			sum += temp[i][j]
	temp = temp/sum
	return temp

def gaussian(img, trows, tcols, variance):
	temp = gaussian_template(trows, tcols, variance)
	return temp_conv(img, temp)
				
if __name__ == "__main__":
	img = imread('lena.png')
	plt.imshow(img)
	plt.show()

	img = greyscale(img)
	plt.imshow(img, cmap = cm.gray)
	plt.show()

	print "Enter the dimensions of template"
	trows = int(raw_input())
	tcols = int(raw_input())

	choice = int(raw_input("Enter operation\n1.Template convolution\n2.Average\n3.Gaussian average"))

	if choice == 1:
		temp = list()
		print "Enter the elements"
		for i in range(trows*tcols):
			temp.append(float(raw_input()))
		temp = np.asarray(temp)
		temp = temp.reshape(trows,tcols)
		newimg = temp_conv(img, temp)
		plt.imshow(newimg, cmap = cm.gray)
		plt.show()
	elif choice == 2:
		newimg = average(img, trows, tcols)
		plt.imshow(newimg, cmap = cm.gray)
		plt.show()
	else:
		variance = float(raw_input("Enter value of variance"))
		newimg = gaussian(img, trows, tcols, variance)
		plt.imshow(newimg, cmap = cm.gray)
		plt.show()
