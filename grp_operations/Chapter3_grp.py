from scipy import misc 
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.557, 0.144])

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

def median_filter(A,win_size):
	rows,cols=A.shape
	C=A.copy()
	B=np.zeros(win_size*win_size)
	for i in range(rows-1):
		for j in range(cols-1):
			count=0
			if(i>(win_size/2)-1 and j>(win_size/2)-1):
				for k in range(win_size/2):
					for l in range(win_size/2):
						B[count]=A[i+k,j+l]
						count=count+1
						B[count]=A[i-k,j-l]
						count=count+1
						B[count]=A[i+k,j-l]
						count=count+1
						B[count]=A[i-k,j+l]
						count=count+1
				C[i,j]=np.median(B)
	return C			
	
def mode_filter(A,win_size):
	rows,cols=A.shape
	C=A.copy()
	B=np.zeros(win_size*win_size)
	for i in range(rows-1):
		for j in range(cols-1):
			count=0
			if(i>(win_size/2)-1 and j>(win_size/2)-1):
				#print(win_size/2)
				for k in range(win_size/2):
					for l in range(win_size/2):
						
						B[count]=A[i+k,j+l]
						count=count+1
						B[count]=A[i-k,j-l]
						count=count+1
						B[count]=A[i+k,j-l]
						count=count+1
						B[count]=A[i-k,j+l]
						count=count+1
				med=np.median(B)
				mean=np.mean(B)
				lower=2*med-np.max(B)
				upper=2*med-np.min(B)
				ListD=[]
				listCount=0
				for k in range(count):
					if(B[k]<upper and med<mean):
						ListD.append(B[k])
						listCount=listCount+1
					if(B[k]>lower and med>mean):
						ListD.append(B[k])
						listCount=listCount+1
				ArrayE=np.asarray(ListD)
				if(listCount>0):
					C[i,j]=np.median(ArrayE)	
				else: C[i,j]=med
	return C				

if __name__=="__main__":
	lena=misc.imread("geoaware1.jpg")
	lena=rgb2gray(lena)
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
	#median filter
	med_lena=median_filter(lena,5)
	plt.imshow(med_lena,cmap=plt.cm.gray)
	plt.show()
	#mode filter
	mod_lena=mode_filter(lena,5)
	plt.imshow(mod_lena,cmap=plt.cm.gray)
	plt.show()
