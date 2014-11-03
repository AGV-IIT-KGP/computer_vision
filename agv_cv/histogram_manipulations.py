import numpy as np


def rgb2grey(image):
    grey = np.floor(np.mean(image, axis=2))
    return grey


def equalise(image):
    rows, cols, channels = image.shape
    if (channels == 3):
        image = rgb2grey(image)
    bright_sum = hist = np.zeros(256)
    psum = 0
    for i in range(0, rows):
        for j in range(0, cols):
            bright_sum[image[i][j]] = bright_sum[image[i][j]] + 1

    number_of_pixels = rows*cols
    for i in range(256):
        psum = bright_sum[i] + psum
        hist[i] = np.floor(255*1.0*psum)/number_of_pixels

    new = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            new[i][j] = hist[image[i][j]]
    return new


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
