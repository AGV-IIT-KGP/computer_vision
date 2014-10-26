import numpy as np 
from scipy.misc import imread
import matplotlib.pyplot as plt
import math

def equalize(img):
	rows,cols,depth=img.shape
	pixels=np.array([0 for x in range(256)])
	for i in range(0,rows):
		for j in range(0,cols):
			bright=img[i,j]
			pixels[bright]=pixels[bright]+1

	sumt=0
	q=np.arange(0,256)
	for level in range(0,255):
		sumt=sumt+pixels[level]
		q[level]=math.floor((255*sumt)/(512*512)+0.00001)
	imgEq=img.copy()
	for x in range(0,rows):
		for y in range(0,cols):
			imgEq[x,y]=q[img[x,y]]
	return imgEq

def otsu(img):
	rows,cols,depth=img.shape
	pixels=np.array([0 for x in range(256)])
	for i in range(0,rows):
		for j in range(0,cols):
			bright=img[i,j]
			pixels[bright]=pixels[bright]+1

	def w(k,pixels):
		sum=0
		for i in range(0,k):
			sum=pixels[i]
		return sum*1.0/(rows*cols)

	def u(k,pixels):
		sum=0
		for i in range(0,k):
			sum=i*pixels[i]
		return sum*1.0/(rows*cols)

	histoT, bin_edges=np.histogram(img,bins=range(256))
	histo=np.array(histoT)
	values_pixels=np.array([0 for i in range(256)])
	values_histo=np.array([0 for i in range(256)])


	for j in range(256):
		if w(j,pixels)==0: continue
		values_pixels[j]=(((u(255,pixels)*w(j,pixels))-u(j,pixels))**2)/((w(j,pixels))*(1-w(j,pixels)))
		if w(j,histo)==0: continue
		values_histo[j]=(((u(255,histo)*w(j,histo))-u(j,histo))**2)/((w(j,histo))*(1-w(j,histo)))

	threshold=np.argmax(values_pixels)
	print "threshold using pixels",threshold
	print "threshold using np.histogram",np.argmax(values_histo)

	from skimage.filter import threshold_otsu
	print "threshold using scikit" , np.floor(threshold_otsu(img))
	
	imgO=np.zeros((rows,cols),int)
	for i in range(cols):
		for j in range(rows):
			if img[i,j,0]>=threshold: 
				imgO[i,j]=255
			elif img[i,j,0]<threshold:
				imgO[i,j]=0
	return imgO

def scikit_otsu(img):
	rows,cols,depth=img.shape
	from skimage.filter import threshold_otsu
	print "threshold using scikit" , np.floor(threshold_otsu(img))
	threshold=np.floor(threshold_otsu(img))
	imgO=np.zeros((rows,cols),int)
	for i in range(cols):
		for j in range(rows):
			if img[i,j,0]>=threshold: 
				imgO[i,j]=255
			elif img[i,j,0]<threshold:
				imgO[i,j]=0
	return imgO

def normalize(img):
	rows,cols,depth=img.shape
	pixels=np.array([0 for x in range(256)])
	for i in range(0,rows):
		for j in range(0,cols):
			bright=img[i,j]
			pixels[bright]=pixels[bright]+1
	mini=np.min(np.min(img))
	maxi=np.max(np.max(img))
	imgRange=maxi-mini

	imgN=img.copy()
	print imgRange
	imgN=np.floor((img-mini)*(255.0/imgRange))
	return imgN

if __name__== "__main__":
	lenag=imread('lenag.bmp')
	plt.imshow(lenag)
	plt.show()
	plt.imshow(otsu(lenag))
	plt.show()
	plt.imshow(scikit_otsu(lenag))
	plt.show()
