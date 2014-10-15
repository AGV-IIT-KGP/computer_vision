import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

def greyscale(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def inversion(img):
	return -img

def addition(img, n):
	aimg = img + n
	np.clip(aimg, 0, 255, out = aimg)
	return aimg

def scaling(img, k):
	simg = k*img
	np.clip(simg, 0, 255, out = simg)
	return simg

def sawtooth(img, slope, step):
	swimg = (img%step)*slope
	np.clip(swimg, 0, 255, out = swimg)
	return swimg

if __name__ == "__main__":
 	image = imread('lena.png')
	plt.imshow(image)
	plt.show()

	img = greyscale(image)
	plt.imshow(img, cmap = cm.gray)
	plt.show()

	while True:
		print "Enter a choice", "1. Inversion", "2. Addition", "3. Scaling" , "4. Sawtooth, Break otherwise"
		choice = int(raw_input())
		if choice == 1:
			inv = inversion(img)
			plt.imshow(inv, cmap = cm.gray)
			plt.show()
		elif choice == 2:
			intercept = int(raw_input("Enter value to be added "))
			added = addition(img, intercept)
			plt.imshow(added, cmap = cm.gray)
			plt.show()
		elif choice == 3:
			slope = int(raw_input("Enter value with which to multiply "))
			scaled = scaling(img, slope)
			plt.imshow(scaled, cmap = cm.gray)
			plt.show()
		elif choice == 4:
			slope = int(raw_input("Enter slope of sawtooth"))
			step = int(raw_input("Enter step value of sawtooth operation"))
			sawn = sawtooth(img, slope, step)
			plt.imshow(sawn, cmap = cm.gray)
			plt.show()
		else:
			break

