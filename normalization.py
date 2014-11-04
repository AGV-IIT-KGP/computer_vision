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

lena = mpimg.imread('lena.bmp') 
gray = rgb2gray(lena)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

plt.hist(gray)
plt.show()

normal = normalize(gray)

plt.hist(normal)
plt.show()
