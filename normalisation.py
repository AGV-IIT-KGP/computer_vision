import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from scipy.misc import imread

def rgb2grey(image):
	grey=np.floor(np.mean(image,axis=2))
	return grey

def normalise(image):
	rows,cols,channels = image.shape
	amax = 0
	amin = 255
	if (channels==3):
		image = rgb2grey(image)
	amax=np.amax(image)
	amin=np.amin(image)
	image[:rows,:cols]=(255-0)/(amax-amin)*image[:rows,:cols]+0
	return image

if __name__ == "__main__":
	panda=imread('panda.jpg')
	norm=normalise(panda)
	plt.imshow(norm,cmap=cm.Greys_r)
	plt.show()
