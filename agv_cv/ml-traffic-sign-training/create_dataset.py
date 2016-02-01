########################################################################################
# Description: Create dataset for training.                
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
#python create_dataset.py "/media/manash/387A7ECD7A7E8780/Documents/Courses/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/0000" "ppm" 4 9

runner, mypath1, myformat, y, no = argv

image_size = 20
threshold = 0.0 
threshold = image_size*image_size*200*0.4

X = [[0 for x in range(400)] for x in range(0)] 
Y = [[0 for x in range(1)] for x in range(0)] 
counter = 0	

for i in range(int(y),int(no)):
	print i
	mypath = mypath1+str(i) 
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	for _file in onlyfiles:
		if (_file[len(_file)-3 : len(_file)] == myformat):
			_image = cv2.imread(mypath+"/" + _file , 1)
			_image = cv2.cvtColor(_image,cv2.COLOR_RGB2GRAY)
			_image = cv2.resize(_image,(image_size,image_size), interpolation = cv2.INTER_CUBIC)
			img_sum = N.sum(_image)
			if(img_sum > threshold ):
				counter = counter+1
				#cv2.imshow("Output",_image)
				#cv2.waitKey(20)
				_image = N.reshape(_image, (N.product(_image.shape)), order ='F')				
				X.append(_image)
				Y.append(int(i)-int(y)+1)
			
		
tel = {'X': X, 'y': Y}
scipy.io.savemat("dataset.mat",tel)
print counter
