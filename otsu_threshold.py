import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imread


def rgb2grey(image):
    grey =np.floor(np.mean (image,axis = 2))
    return grey


def otsu(image):
    rows, cols, channels = image.shape
    if (channels == 3):
        image = rgb2grey(image)
    hist=np.zeros(256)
    hist,bin_edges = np.histogram(image,bins=256,range=(0,255))
    w=np.zeros(256)
    mu=np.zeros(256)
    N=rows*cols
    w[0]=(hist[0]*1.0)/(N)
    for i in range(255):
        w[i+1] = w[i]+(hist[i+1]*1.0)/N
    mu[0]=0
    for i in range(255):
        mu[i+1] = mu[i]+(i*hist[i]*1.0)/(N)
    result=np.array([0 for i in range(256)])
    for i in range(256):
        t=(mu[255]*w[i]-mu[i])
        if (w[i] == 0): continue
        if (w[i] == 1):continue
        result[i]=t*t/(w[i]*(1-w[i]))
    max=result[0]
    index=0
    index=np.argmax(result)
    return index

if __name__ == "__main__":
    panda=imread('lena.bmp')
    g = rgb2grey(panda)
    rows,cols,channels = panda.shape
    index = otsu(panda)
    print(index)
    for i in range(rows):
        for j in range(cols):
            if (g[i][j] > index):
                g[i][j] = 255
            if (g[i][j] <= index):
                g[i][j]=0
    plt.imshow(g,cmap=cm.Greys_r)
    plt.show()

