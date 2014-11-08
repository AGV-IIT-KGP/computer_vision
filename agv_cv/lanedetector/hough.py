import numpy as np
import math

def hough_transform(img_bin, num_lines=2, theta_res=1, rho_res=1):
    nR,nC = img_bin.shape
    theta = np.linspace(0.0, 180.0, np.ceil(180/theta_res) + 1.0)
    D = np.sqrt((nR - 1)**2 + (nC - 1)**2)
    q = np.ceil(D/rho_res)
    nrho = 2*q + 1
    rho = np.linspace(0.0, nrho, nrho + 1.0)
    H = np.zeros((len(rho), len(theta)))
    for rowIdx in range(nR):
        for colIdx in range(nC):
            if img_bin[rowIdx, colIdx]:        #for lines of white pixels in a black background, use 'not' for black lines in white background
                for thIdx in range(len(theta)):
                    rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) + rowIdx*np.sin(theta[thIdx]*np.pi/180)
                    rhoVal = math.floor(rhoVal)
                    H[rhoVal, thIdx] += 1

    p = np.zeros(num_lines)
    th = np.zeros(num_lines)
    for i in range(num_lines):
        biggest = np.argmax(H)
        p[i], th[i] = np.unravel_index(biggest, H.shape)
        H[p[i]][th[i]] = 0
    newimg = np.zeros((img_bin.shape))
    for i in range(nR):
        for k in range(num_lines):
            if th[k] != 0:
                j = p[k]/math.sin(th[k]*np.pi/180) - i*math.cos(th[k]*np.pi/180)/math.sin(th[k]*np.pi/180)
                if j<nC:
                    if j<0:
                        j = -j
                    newimg[i][j] = 255
    return newimg
