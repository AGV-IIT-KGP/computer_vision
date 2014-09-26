import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

img = imread('l.png')
plt.imshow(img)
plt.show()

def rgb2grey(pixel):
    return math.floor((0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]) / 3)

#creating a greyscale version
grey = np.zeros((img.shape[0], img.shape[1])) 
for rownum in range(len(img)):
   for colnum in range(len(img[rownum])):
      grey[rownum][colnum] = rgb2grey(img[rownum][colnum])

plt.imshow(grey, cmap = cm.Greys_r)
plt.show()

plt.hist(img.flatten())
plt.show()

plt.hist(grey)
plt.show()

[rows, cols] = grey.shape
print rows
print cols

max = np.amax(grey)
print max
min = np.amin(grey)
print min

width  = max - min

#normalizing the greyscale image
normal = np.zeros((grey.shape[0], grey.shape[1]))
for y in range(rows):
	for x in range(cols):
		normal[y][x] = math.floor((grey[y][x] -  min)*255/width)

plt.imshow(normal)
plt.show()

plt.hist(normal)
plt.show()