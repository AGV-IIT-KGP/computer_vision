import numpy as np 
import math

def convolve(img,template):
    rowsI,colsI,depthI=img.shape
    rowsT,colsT=template.shape
    imgC=img.copy()
    
    for i in range(rowsT/2+1,rowsI-rowsT/2):
        for j in range(colsT/2+1,colsI-colsT/2):
            sum=0
            for x in range(rowsT):
                for y in range(colsT):
                    sum=sum+template[x,y]*img[i-rowsT/2+x,j-colsT/2+y]
            imgC[i,j]=sum
    return imgC

def mean(img,temp_size):
    template=np.zeros((temp_size,temp_size),float)+1.0/(temp_size**2)
    return convolve(img,template)

def guass_mean(img,temp_size,sigma):
    template=np.zeros((temp_size,temp_size),float)
    centre=np.floor(temp_size/2)+1
    sum=0
    for i in range(temp_size):
        for j in range(temp_size):
            template[i,j]=np.exp(-(((j-centre)**2)+((i-centre)**2))/(2*sigma**2))
            sum=sum+template[i,j]
    template=template/sum
    return convolve(img,template)

def medianSQR(img,temp_size):
    rowsI,colsI,depthI=img.shape
    imgMSq=img.copy()
    for x in range(temp_size/2+1,rowsI-temp_size/2):
        for y in range(temp_size/2+1,colsI-temp_size/2):
            temp=[]
            for i in range(temp_size):
                for j in range(temp_size):
                    temp=temp+[img[i-temp_size/2+x,j-temp_size/2+y]]
            sorted=[]
            sorted=np.sort(temp)
            imgMSq[x,y]=sorted[temp_size/2]
    return imgMSq


   ''' def trunc_med(img,temp_size):
    imgTM=img.copy()
    rowsI,colsI,depthI=img.shape
    half=(temp_size/2)
    for x in range(half,colsI-half-1):
        for y in range(half,rowsI-half-1):
            window=img[x-half:x+half,y-half:y+half]
            t_ave=np.mean(window.flatten())
            t_med=np.median(window.flatten())
            upper=2*t_med-np.min(window)
            lower=2*t_med-np.max(window)
            cc=0
            trun=[]
            for i in range(0,temp_size):
                for j in range(0,temp_size):
                    if ((window[i,j]<upper) and (t_med<t_ave)):
                        trun[cc]=window[i,j]
                        cc+=1
                    if ((window[i,j]>lower) and (t_med>t_ave)):
                        trun[cc]=window[i,j]
                        cc+=1
            if cc>0:
                imgTM[x,y]=np.median(trun)
            else:
                imgTM[x,y]=t_med
    return imgTM
    '''
    

