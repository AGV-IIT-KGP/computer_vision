import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imread
import scipy.signal as sig
import math

def rgb2grey(image):
	rows,cols,channels= image.shape
	if (channels == 3):
		image = np.mean(image,axis=2)
	return image

def normalise(image):
	if (len(image.shape)==3):
		image = rgb2grey(image)
	rows,cols = image.shape
	amax=np.amax(image)
	amin=np.amin(image)
	image[:rows,:cols]=(2000-0)/(amax-amin)*image[:rows,:cols]+0
	return image


def convolve(matA,matB):
	rows,cols= matA.shape
	rowB,colB= matB.shape
	result = np.zeros((rows,cols))
	center = np.floor(rowB/2)
	no = center * center  
	for i in range(int(center),int(rows-center)):
		for y in range(int(center),int(cols-center)):
			sum = matA[i-center:i+center+1,y-center:y+center+1]*matB[center-center:center+center+1,center-center:center+center+1]
			sum = np.sum(sum)/no
			result[i,y] = sum
	result = normalise(result)
	return result



def average(image,winsize):
	image = rgb2grey(image)
	rows,cols = image.shape
	no_of_pixels = winsize * winsize
	convolvemat = np.ones((winsize,winsize))
	convolvemat = np.divide(convolvemat,no_of_pixels)
	image = convolve(image,convolvemat)
	return image

def guassian(image,winsize,sigma):
	center =int(winsize/2)
	convolvemat = np.zeros((winsize,winsize))
	for i in range(0,winsize):
		for j in range(0,winsize):
			convolvemat[i,j] = math.exp(-((j-center)*(j-center)+(i-center)*(i-center))/2/sigma/sigma)
	convolvemat = np.divide(convolvemat,np.sum(convolvemat))
	image = rgb2grey(image)
	image = convolve(image,convolvemat)
	return image

if __name__ == "__main__":
	lena = imread('lena.bmp')
	lena1 = average(lena,3)
	plt.imshow(lena1,cmap=cm.Greys_r)
	plt.show()
	lena2 = guassian(lena,5,1)
	plt.imshow(lena2,cmap=cm.Greys_r)
	plt.show()
