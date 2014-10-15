import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from scipy.misc import imread
panda=imread('panda.jpg')

def normalise(image):
	rows,cols,channels = image.shape
	amax = 0
	amin = 255
	if (channels == 3):
		grey = np.mean(image,axis=2)
	amax=np.amax(grey)
	amin=np.amin(grey)
	grey[:rows,:cols]=(255-0)/(amax-amin)*grey[:rows,:cols]+0
	return grey

if __name__ == "__main__":
	panda = imread('panda.jpg')
	grey=normalise(panda)
	plt.imshow(grey,cmap=cm.Greys_r)
	plt.show()
