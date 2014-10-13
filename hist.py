import numpy as np 
from scipy.misc import imread
import matplotlib.pyplot as plt
lenag=imread('lenag.bmp')

a=np.arange(0,256)
pixels=np.array([0 for x in range(256)])
for i in range(0,512):
	for j in range(0,512):
		bright=lenag[i,j]
		pixels[bright]=pixels[bright]+1

plt.plot(a,pixels)
plt.show()