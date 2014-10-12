import numpy as np 
from scipy.misc import imread
import matplotlib.pyplot as plt
import math 

lenag=imread('lenag.bmp')
rows, cols, depth=lenag.shape
a=np.arange(0,256)
pixels=np.array([0 for p in range(256)])
for x in range(0,rows):
	for y in range(0,cols):
		bright=lenag[x,y]
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

values=np.array([0 for i in range(256)])

for j in range(256):
	if w(j,pixels)==0: continue
	values[j]=(((u(255,pixels)*w(j,pixels))-u(j,pixels))**2)/((w(j,pixels))*(1-w(j,pixels)))

threshold=np.max(values)

lenagO=lenag.copy()

for i in range(cols):
	for j in range(rows):
		if lenag[i,j,0]>=threshold: 
			lenagO[i,j]=255
		elif lenag[i,j,0]<threshold:
			lenagO[i,j]=0

plt.imshow(lenagO)
plt.xlabel('Otsu thresholding')
plt.show()

