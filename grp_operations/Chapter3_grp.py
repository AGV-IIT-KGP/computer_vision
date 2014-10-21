from scipy import misc 
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np

def normalisation(gray,nmin,nmax):
	omax=np.amax(lena)
	omin=np.amin(lena)
	conv=(nmax-nmin)*1.0/(omax-omin)
	lena2=gray.copy()
	lena2=conv*(lena-omin) + nmin
	return lena2		

def convolution(A,B):
	result=sig.convolve2d(A,B,'valid','fill',0)
	return result

def average(A,win_size):
	winsum=win_size*win_size
	B=np.ones((win_size,win_size))
	B=np.divide(B,winsum)
	return convolution(A,B)

def gaussian_avg(A,win_size,sigma):
	B=np.ones((win_size,win_size))
	center=np.floor(win_size/2)
	for i in range(win_size):
		for j in range(win_size):
			B[i,j]=-(((i-center)*(i-center)+(j-center)*(j-center))/(2*sigma*sigma))
	B=np.exp(B)
	asum=np.sum(B)
	B=np.divide(B,asum)
	#print(B)
	return convolution(A,B)		

if __name__=="__main__":
	lena=misc.imread("lena.png")
	rows,cols=lena.shape
	print(rows,cols)
	#convolution
	B=np.array([[1,1,1],[1,1,1],[1,1,1]])
	convolve_lena=convolution(lena,B)
	plt.imshow(convolve_lena,cmap=plt.cm.gray)
	plt.show()
	#direct averaging
	avg_lena=average(lena,5)
	plt.imshow(avg_lena,cmap=plt.cm.gray)
	plt.show()
	#gaussian averaging
	gaussian_lena=gaussian_avg(lena,5,1)
	plt.imshow(gaussian_lena,cmap=plt.cm.gray)
	plt.show()
