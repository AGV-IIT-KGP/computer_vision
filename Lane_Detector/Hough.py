import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from skimage import filter

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def getbinarylanes(img):
    threshold = filter.threshold_otsu(img)
    img[img>=threshold] = 255
    img[img<threshold] = 0
    return img

def hough_transform(img_bin, num_lines=2, theta_res=1, rho_res=1):
    nR,nC = img_bin.shape
    theta = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1.0)
    theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
     
    D = np.sqrt((nR - 1)**2 + (nC - 1)**2)
    q = np.ceil(D/rho_res)
    nrho = 2*q + 1
    rho = np.linspace(-q*rho_res, q*rho_res, nrho)
    H = np.zeros((len(rho), len(theta)))
    for rowIdx in range(nR):
        for colIdx in range(nC):
            if img_bin[rowIdx, colIdx]:        #for lines of white pixels in a black background, use 'not' for black lines in white background
                for thIdx in range(len(theta)):
                    rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) + \
                    rowIdx*np.sin(theta[thIdx]*np.pi/180)
                rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
                H[rhoIdx[0], thIdx] += 1

    p = np.zeros(num_lines)
    th = np.zeros(num_lines)
    for i in range(num_lines):
        p[i] = np.argmax(np.max(H, axis = 0))
        th[i] = np.argmax(np.max(H,axis =1))
        H[th[i]][p[i]] = 0
    
    newimg = np.zeros((img_bin.shape))
    for i in range(nR):
        for k in range(num_lines):
            j = p[k]/math.sin(th[k]*np.pi/180) - i*math.cos(th[k]*np.pi/180)/math.sin(th[k]*np.pi/180)
            if j<nC and j>=0:
                newimg[i][j] = 255
    return newimg

if __name__ == "__main__":
    colorimg = imread('sample1.jpeg')
    plt.imshow(colorimg)
    plt.show()

    img = rgb2gray(colorimg)
    plt.imshow(img, cmap = cm.gray)
    plt.show()
    
    img = getbinarylanes(img)
    plt.imshow(img, cmap = cm.gray)
    plt.show()

    newimg = hough_transform(img, 50)
    plt.imshow(newimg, cmap = cm.gray)
    plt.show()

