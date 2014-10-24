import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imread

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
def normalize(gray):
	normal = np.floor(((gray - np.amin(gray))*255)/(np.amax(gray)-np.amin(gray)))
	return normal
def convolute(img,temp):
	irows, icols = img.shape
	trows, tcols = temp.shape
	elsum = np.sum(temp)
			
	final = np.array([0.0 for i in range(irows * icols)])
	final = final.reshape(irows, icols)
	
	trhalf = trows/2
	tchalf = tcols/2
	
	for i in range(trhalf, irows - trhalf):
		for j in range(tchalf, icols - tchalf):
			final[i][j] = np.floor(np.sum(img[i-trhalf:i+trhalf+1, j-tchalf:j+tchalf+1] * temp)/elsum)
	
	return final

def gaussian_template(size, sigma):
	centre = size/2 + 1
	template = np.array([0.0 for i in range(size*size)])
	template = template.reshape(size,size)
	tsum = 0
	for i in range(1,size+1):
		for j in range(1,size+1):
			template[i-1][j-1] = np.exp(-(((j-centre)*(j-centre))+((i-centre)*(i-centre)))/(2.0*sigma*sigma))
			tsum += template[i-1][j-1]
	template = template/tsum
	return template







if __name__ == "__main__":
	lena = mpimg.imread('lena.bmp') 
	gray = rgb2gray(lena)
	gray = normalize(gray)    
	plt.imshow(gray, cmap = plt.get_cmap('gray'))
	plt.show()

	print "Enter one of the following.\n1.Template Convolution\n2.Averaging Operator\n3.Gaussian Operator"
	ch = int(raw_input())
	print "Enter the template dimensions."
	trows = int(raw_input())
	tcols = int(raw_input())

	if ch == 1:
		template = np.array([0.0 for i in range(trows * tcols)])
		for i in range (trows * tcols):
			print "Enter weight."
			template[i] = float(raw_input())
		template = template.reshape(trows,tcols)
		image = convolute(gray, template)
		plt.imshow(image, cmap='gray')
		plt.show()
	elif ch  ==2:
		template = np.array([1.0 for i in range(trows * tcols)])
		template = template.reshape(trows,tcols)
		image = convolute(gray, template)
		plt.imshow(image, cmap='gray')
		plt.show()
	else:
		print "Enter sigma."
		s = float(raw_input())
		template = gaussian_template(trows,s)
		image = convolute(gray, template)
		plt.imshow(image, cmap='gray')
		plt.show()
		