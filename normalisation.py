import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from scipy.misc import imread
panda=imread('panda.jpg')
plt.imshow(panda)
plt.imshow(panda)
rows,cols,channels = panda.shape
print(panda.shape)
amax = 0
amin = 255
grey = np.mean(panda,axis=2)
#for i in range(rows):
 #   for j in range(cols):
  #      if (amax > panda[i,j,0]):
   #         amax = panda[i,j,0]
    #    if (amin < panda[i,j,0]):
     #      amin = panda[i,j,0]
amax=np.amax(grey)
amin=np.amin(grey)
plt.imshow(grey,cmap=cm.Greys_r)
grey[:rows,:cols]=(255-0)/(amax-amin)*grey[:rows,:cols]+0
print(grey.flatten())
plt.hist(grey.flatten(),normed=True)
