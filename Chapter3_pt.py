from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.557, 0.144])


lena=misc.imread("geoaware1.jpg")
lena=rgb2gray(lena)
print(lena.shape)
plt.imshow(lena,cmap=plt.cm.gray)
plt.show()
hist=np.array([0 for i in range(256)])	
rows,cols = lena.shape
nmin=0
nmax=255
new_range=nmax-nmin
number_of_pixels=rows*cols
omax=lena[0][0]
omin=lena[0][0]
for i in range(rows):
	for j in range(cols):
		if(lena[i,j]>omax): omax=lena[i,j]
		if(lena[i,j]<omin): omin=lena[i,j]
print(omax,omin)
print(nmax,nmin)

#histogram creation....
for i in range(rows):
	for j in range(cols):
		hist[lena[i,j]]+=1
plt.hist(lena.flatten())
plt.show()
plt.plot(hist)
plt.show()

#histogram normalisation... to 50->200

conv=(nmax-nmin)*1.0/(omax-omin)
print("conv",conv)
lena2=lena.copy()
for i in range(rows):
	for j in range(cols):
		lena2[i,j]=conv*(lena[i,j]-omin) + nmin
plt.imshow(lena2,cmap=plt.cm.gray)
plt.show()


#histogram equalisation

psum=0
hist2=np.array([0 for i in range(256)])
for i in range(256):
	psum+=hist[i]
	hist2[i]=(new_range*1.0*psum)/number_of_pixels
lena3=lena.copy()
for i in range(rows):
	for j in range(cols):
		lena3[i,j]=hist2[lena[i,j]]
plt.imshow(lena3,cmap=plt.cm.gray)
plt.show()
plt.hist(lena3.flatten())
plt.show()


#OTSU thresholding

w=np.array([0.00 for i in range(256)])
w[0]=(hist[0]*1.0)/number_of_pixels
for i in range(255):
	w[i+1]=w[i]+(hist[i+1]*1.0)/number_of_pixels 

mu=np.array([0 for i in range(256)])
mu[0]=1*hist[0]*1.0/number_of_pixels
for i in range(255):
	mu[i+1]=mu[i]+((i+2)*hist[i+1]*1.0)/number_of_pixels

muT=mu[255]

#plt.plot(w)
#plt.show()
thresh=0
maxValue=0
for i in range(256):
	if(w[i]==0):continue
	val=(muT*w[i]-mu[i])*(muT*w[i]-mu[i])/(w[i]*(1.0-w[i]))
	if(val>maxValue):
		maxValue=val
		thresh=i

print(thresh)
lena4=lena.copy()
for i in range(rows):
	for j in range(cols):
		if(lena[i,j]>=thresh):lena4[i,j]=255;
		else : lena4[i,j]=0;

plt.imshow(lena4,cmap=plt.cm.gray)
plt.show()
