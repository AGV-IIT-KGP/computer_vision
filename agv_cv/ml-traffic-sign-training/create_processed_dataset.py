########################################################################################
# Description: Create dataset for training after applying processing to each image.               
# Input: image files of a particular format as passed through arguments.           
# Output: "dataset.mat" with X and Y of the dataset                
# Dataset Format: X is a (a,b) matrix. a is the number of images, while b 
#                   is the total number of pixels in the image.
#                 Y is a (a,1) matrix that defines the class each image in X belongs to.
# Author: Manash Pratim Das (mpdmanash@gmail.com)
########################################################################################

import cv2
import numpy as N
from sys import argv
import scipy.misc, scipy.io, scipy.optimize, scipy.special

from os import listdir
from os.path import isfile, join

#USAGE
#python create_processed_dataset.py "/media/manash/387A7ECD7A7E8780/Documents/Courses/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/0000" "ppm" 4 9

runner, mypath1, myformat, y, no = argv

image_size = 28
threshold = 0.0 
threshold = image_size*image_size*100*0.1

X = [[0 for x in range(image_size*image_size)] for x in range(0)] 
Y = [[0 for x in range(1)] for x in range(0)] 
counter = 0	

for i in range(int(y),int(no)):
	print i
	mypath = mypath1+str(i) 
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	for _file in onlyfiles:
		if (_file[len(_file)-3 : len(_file)] == myformat):
			_image = cv2.imread(mypath+"/" + _file , 1)

			if(_image.shape[0] > 50):				
				_image = cv2.cvtColor(_image,cv2.COLOR_RGB2GRAY)

				_image = cv2.Canny(_image,100,200)

				kernel = N.ones((3,3),N.uint8)

				#_image = cv2.dilate(_image,kernel,iterations = 1) 
				#_image = cv2.dilate(_image,kernel,iterations = 1)
				#_image = cv2.erode(_image,kernel,iterations = 1)
				#_image = cv2.erode(_image,kernel,iterations = 1)
				#cv2.imshow("Output",_image)
				#cv2.waitKey(0)

				mix = _image.shape[0]/2
				flow = _image.shape[0]*0.25

				img_sum = N.sum(_image[mix-flow:mix+flow,mix-flow:mix+flow])

				_image = cv2.resize(_image,(image_size,image_size), interpolation = cv2.INTER_CUBIC)
				
				#print img_sum
				if(img_sum > threshold ):
					counter = counter+1
					#cv2.imshow("Output",_image)
					#cv2.waitKey(0)
					_image = N.reshape(_image, (N.product(_image.shape)), order ='F')				
					X.append(_image)
					Y.append(int(i)-int(y)+1)
			
		
tel = {'X': X, 'y': Y}
scipy.io.savemat("dataset.mat",tel)
print counter






