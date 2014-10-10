import numpy as np 
from scipy.misc import imread
import matplotlib.pyplot as plt
lenag=imread('lenag.bmp')


def histo(image,label):
	a=np.arange(0,256,1)
	b=[0]
	c=[0]

	for i in range(0,255):
		b=b+c
	pixels=np.array(b)
	for bright in a:
		pixels[bright]=0
	for x in range(0,512):
		for y in range(0,512):
			bright=image[x,y]
			pixels[bright]=pixels[bright]+1
	plt.plot(a,pixels)
	plt.xlabel(label)
	plt.show()
	return

histo(lenag,'Normal Histogram')

lenagN=((lenag-25)*(256/175))

histo(lenagN,'Normalized Histogram')








