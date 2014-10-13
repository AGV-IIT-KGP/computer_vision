import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from scipy.misc import imread
panda = imread('lena.bmp')
def rgb2grey(image):
    grey =np.floor(np.mean (image,axis = 2))
    return grey
g = rgb2grey(panda)
bright_sum=hist= np.zeros((256,1))
psum=0
rows,cols,channels = panda.shape
for i in range(0,rows):
    for j in range(0,cols):
        bright_sum[g[i][j]]=bright_sum[g[i][j]] + 1
number_of_pixels=rows*cols
for i in range(256):
    psum =bright_sum[i] + psum
    hist[i]=np.floor(255*1.0*psum)/number_of_pixels
new = np.zeros((rows,cols))
for i in range(rows):
    for j in range(cols):
        new[i][j]=hist[g[i][j]]
plt.hist(new.flatten())
plt.show()
