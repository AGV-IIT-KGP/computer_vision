import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imread

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

lena = mpimg.imread('lena.bmp') 
gray = rgb2gray(lena)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()
rows,cols = gray.shape

hist=np.array([0 for i in range(256)])
eqhist=np.array([0 for i in range(256)])
for i in range(rows):
	for j in range(cols):
		hist[gray[i,j]]+=1
sum=0
for i in range(256):
	sum+=hist[i]
	eqhist[i] = (255.0*sum)/(rows*cols)
eq = gray.copy()
for i in range(rows):
	for j in range(cols):
		eq[i,j]=eqhist[gray[i,j]]


plt.imshow(eq,cmap = plt.get_cmap('gray'))
plt.show()
plt.hist(eq.flatten())
plt.show()