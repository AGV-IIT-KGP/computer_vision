import numpy as np 
from scipy.misc import imread
import matplotlib.pyplot as plt
import math 

lenag=imread('lenag.bmp')
a=np.arange(0,256)
b=[0]
c=[0]

for i in range(0,255):
	b=b+c
pixels=np.array(b)
for bright in a:
	pixels[bright]=0
for x in range(0,512):
	for y in range(0,512):
		bright=lenag[x,y]
		pixels[bright]=pixels[bright]+1

sumt=0
q=np.arange(0,256)
for level in range(0,255):
	sumt=sumt+pixels[level]
	q[level]=math.floor((255*sumt)/(512*512)+0.00001)

plt.plot(a,q)
plt.xlabel('Q vs P')
plt.show()

lenagE=lenag.copy()
for x in range(0,512):
	for y in range(0,512):
		lenagE[x,y]=q[lenag[x,y]]

plt.hist(lenagE.flatten(),256)
plt.xlabel('Equalized Lena')
plt.show()

plt.hist(lenag.flatten(),256)
plt.xlabel('Lena')
plt.show()

plt.imshow(lenagE)
plt.xlabel('Equalized Lena')
plt.show()

plt.imshow(lenag)
plt.xlabel('Lena')
plt.show()
