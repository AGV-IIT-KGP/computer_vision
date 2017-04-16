import numpy as np
import math
from Hough import *

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def normalize(gray):
    normal = np.floor(((gray - np.amin(gray))*255)/(np.amax(gray)-np.amin(gray)))
    return normal

def otsu_threshold(gray):
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
    return k

def thresholding(img,thresh):
    img[img<thresh]=0
    img[img>=thresh]=255
    return img

def kernel_iteration(img, thresh):
    trows, tcols = 3,3
    irows, icols = img.shape
    template = np.array([1.0 for i in range(trows *trows)])
    template = template.reshape(trows,tcols)
    elsum = np.sum(template)

    final = np.array([0.0 for i in range(irows * icols)])
    final = final.reshape(irows, icols)
    
    trhalf = trows/2
    tchalf = tcols/2
    
    for i in range(trhalf, irows - trhalf):
        for j in range(tchalf, icols - tchalf):
            if ((np.sum(img[i-trhalf:i+trhalf+1, j-tchalf:j+tchalf+1] * template)/elsum)<thresh):
                final[i][j]=0
            else:
                final[i][j]=255
    return final

def equalize(gray):
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
    return eq

def gaussian_template(size, sigma):
    centre = size/2 + 1
    template = np.array([0.0 for i in range(size*size)])
    template = template.reshape(size,size)
    tsum = 0
    for i in range(1,size+1):
        for j in range(1,size+1):
            template[i-1][j-1] = np.exp(-(((j-centre)*(j-centre))+((i-centre)*(i-centre)))/(2.0*sigma*sigma))
            tsum += template[i-1][j-1]
    template = template/tsum
    return template

def gaussian_blur(img):
    temp = gaussian_template(5,1)

    irows, icols = img.shape
    trows, tcols = temp.shape
    elsum = np.sum(temp)
            
    final = np.array([0.0 for i in range(irows * icols)])
    final = final.reshape(irows, icols)
    
    trhalf = trows/2
    tchalf = tcols/2
    
    for i in range(trhalf, irows - trhalf):
        for j in range(tchalf, icols - tchalf):
            final[i][j] = np.floor(np.sum(img[i-trhalf:i+trhalf+1, j-tchalf:j+tchalf+1] * temp)/elsum)
    
    return final
