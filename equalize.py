import numpy as np 
from scipy.misc import imread
import matplotlib.pyplot as plt
import math

def equalize(img):
	rows,cols=img.shape
	pixels=np.array([0 for x in range(256)])
	for i in range(0,rows):
		for j in range(0,cols):
			bright=img[i,j]
			pixels[bright]=pixels[bright]+1

	sumt=0
	q=np.arange(0,256)
	for level in range(0,255):
		sumt=sumt+pixels[level]
		q[level]=math.floor((255*sumt)/(512*512)+0.00001)
	imgEq=img.copy()
	for x in range(0,rows):
		for y in range(0,cols):
			imgEq[x,y]=q[img[x,y]]
	return imgEq

lena=imread('lena.bmp')
lenaR=lena[:,:,0].copy()
lenaG=lena[:,:,1].copy()
lenaB=lena[:,:,2].copy()

lenaR_e=equalize(lenaR)
lenaG_e=equalize(lenaG)
lenaB_e=equalize(lenaB)

lena_e=lena.copy()

lena_e[:,:,0]=lenaR_e	
lena_e[:,:,1]=lenaG_e
lena_e[:,:,2]=lenaB_e