import numpy as np 
from scipy.misc import imread
import matplotlib.pyplot as plt
import math

def convolve(img,template):
	rowsI,colsI,depthI=img.shape
	rowsT,colsT=template.shape
	imgC=img.copy()
	
	for i in range(rowsT/2+1,rowsI-rowsT/2):
		for j in range(colsT/2+1,colsI-colsT/2):
			sum=0
			for x in range(rowsT):
				for y in range(colsT):
					sum=sum+template[x,y]*img[i-rowsT/2+x,j-colsT/2+y]
			imgC[i,j]=sum
	return imgC

def mean(img,temp_size):
	template=np.zeros((temp_size,temp_size),float)+1.0/(temp_size**2)
	return convolve(img,template)

def medianSQR(img,temp_size):
	rowsI,colsI,depthI=img.shape
	imgMSq=img.copy()
	for x in range(temp_size/2+1,rowsI-temp_size/2):
		for y in range(temp_size/2+1,colsI-temp_size/2):
			temp=[]
			for i in range(temp_size):
				for j in range(temp_size):
					temp=temp+[img[i-temp_size/2+x,j-temp_size/2+y]]
			sorted=[]
			sorted=np.sort(temp)
			imgMSq[x,y]=sorted[temp_size/2]
	return imgMSq

lenag=imread('lenag.bmp')

plt.imshow(medianSQR(lenag,5))
plt.show()
