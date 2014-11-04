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


hist=np.array([0.0 for i in range(256)])
ots=np.array([0.0 for i in range(256)])

for i in range(rows):
	for j in range(cols):
		hist[gray[i,j]]+=1
hist = (hist*1.0)/(rows * cols)

T = 0
for i in range(256):
	T += ((i+1)*hist[i])
min = np.floor(np.amin(gray))
w = 0
u = 0
k = 0
for i in range(int(min), 256):
	w+= hist[i]
	u+= (i + 1) * hist[i]
	if(w == 1.0):
		continue
	ots[i] = ((T * w - u)**2) / (w * (1 - w))
	if ots[k]<ots[i]:
		k=i;
gray[gray<k]=0
gray[gray>=k]=255
plt.imshow(gray, cmap='gray')
plt.show()