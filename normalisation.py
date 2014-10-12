import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from scipy.misc import imread
panda=imread('panda.jpg')
rows,cols,channels = panda.shape
amax = 0
amin = 255
grey = np.mean(panda,axis=2)
amax=np.amax(grey)
amin=np.amin(grey)
grey[:rows,:cols]=(255-0)/(amax-amin)*grey[:rows,:cols]+0
plt.hist(grey.flatten(),normed=True)
