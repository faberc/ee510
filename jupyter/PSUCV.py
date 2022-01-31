import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy import signal
import scipy

plt.rcParams['text.usetex'] = True

def AddColorbar(axes,colorMap='gray'):
    images = axes.get_images()
    if len(images) > 0:
       vmin, vmax = images[0].get_clim()
    sm = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(vmin=vmin,vmax=vmax),cmap=colorMap)
    bb = np.array(axes.get_position().bounds)
    bb[0] += bb[2]*0.95
    bb[2] *= 0.05
    
    figure = axes.get_figure()
    barAxes = figure.add_axes(bb)
    figure.colorbar(sm,barAxes)

    
def DisplaySideBySide(imageInput,imageOutput):
    figure = plt.figure()
    figure.clf()
    figure.set_size_inches(15,10)

    plt.subplot(121)
    plt.imshow(imageInput)
    plt.title('Input')

    plt.subplot(122)
    plt.imshow(imageOutput)
    plt.title('Output')
    
def GetTestImage(nPixelsPerSide=25):
    iHalf = int(nPixelsPerSide/2)
    i14   = int(nPixelsPerSide*1/4)
    i34   = int(nPixelsPerSide*3/4)
    #image = np.ones((nPixelsPerSide,nPixelsPerSide),np.float32)*0.2
    image = np.zeros((nPixelsPerSide,nPixelsPerSide),np.float32)
    image[iHalf,:] = 1.0
    image[:,iHalf] = 1.0
    image[iHalf:,i34] = 1.0
    image[i34,iHalf:] = 1.0
    image[i14,i34] = 1.0
    image[i34:,iHalf:i34] = 1.0
    for c in range(iHalf): 
        image[c,c] = 1.0
        image[-1-c,c] = 1.0
    return(image)
def GetImpulseImage(nPixelsPerSide=25):
    iHalf = int(nPixelsPerSide/2)
    image = np.zeros((nPixelsPerSide,nPixelsPerSide),np.float32)
    image[iHalf,iHalf] = 1.0
    return(image)
def GetChirpImage(nPixelsPerSide=51,fMax=0.4):
    iv = np.arange(nPixelsPerSide) 
    iv = iv.reshape(-1,1)
    irows = iv*np.ones((1,nPixelsPerSide))
    icols = np.transpose(irows)
    image = signal.chirp(irows,0,irows.max(),fMax)*signal.chirp(icols,0,icols.max(),fMax)
    image = 0.5*image + 0.5
    image = image.astype(np.float32) # OpenCV can't handle float64
    return(image)
def GetDemoImages():
    images = list()
    images.append(GetImpulseImage())
    images.append(GetTestImage())
    images.append(GetChirpImage())
    images.append(cv.imread('DarkTree.jpeg',cv.IMREAD_GRAYSCALE)[:151,:151])
    images.append(cv.imread('Bubble.jpeg',cv.IMREAD_GRAYSCALE)[1001:2000,251:1250])
    scaledImages = list()
    for image in images: # Convert to standard scale of 0-1
        im = image.copy().astype(np.float32)
        im = im/im.max() 
        scaledImages.append(im)
    return(scaledImages)
def GetMinEigenvalues(image,blockSize=3):
    scale  = 1
    delta  = 0
    kSize  = 3 # Must be 1, 3, 5, or 7
    ddepth = cv.CV_64F
    
    imageDx  = cv.Sobel(image, ddepth, 1, 0, ksize=kSize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    imageDy  = cv.Sobel(image, ddepth, 0, 1, ksize=kSize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    A = imageDx**2
    B = imageDy**2
    C = imageDx*imageDy

    kernel = (blockSize,blockSize)
    A = cv.GaussianBlur(A,kernel,0)
    B = cv.GaussianBlur(B,kernel,0)
    C = cv.GaussianBlur(C,kernel,0)

    lambdaMin = abs(((A+B) - np.sqrt((A-B)**2 + 4*C**2))/2)       
    return(lambdaMin)
def Kernel(t,t0,kernelType='Sinc',parameter=np.nan):
    td = t - t0
    if kernelType=='Sinc':
        x = np.sinc(td)
    elif kernelType=='Windowed Sinc':
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html#scipy.signal.windows.get_window
        # for a list of over a dozen different windows that might be used. However, these are discrete-time
        # windows and this function has to work for real-valued values, so you'll need to code in whatever
        # window you want to use yourself
        windowDuration = 5
        window = (1-(td/windowDuration)**2)**2*(np.abs(td)<windowDuration) # Biweight window    
        x = np.sinc(td)*window
    elif kernelType=='Gaussian': # Linear interpolation
        sigma = 0.5425 # Closely matches that used by the binomial of pyrUp and pyrDown in OpenCV
        if not np.isnan(parameter):
            sigma = parameter
        k = int(np.ceil(np.abs(td).min()))
        n = np.arange(-k,k+1)
        xn = np.exp(-(n**2.0)/(2*sigma**2.0))
        x = np.exp(-(td**2.0)/(2*sigma**2.0))/sum(xn)
    elif kernelType=='Tent': # Linear interpolation
        a = -0.5
        x = (1-np.abs(td))*(np.abs(td)<1) 
    elif kernelType=='Binomial3': # Binomial with Piecewise Linear Interpolation
        #  0  1  2  1  0 
        # -2 -1  0  1  2
        us = 1
        u0 = (us*td+2)
        u1 = (us*td+1)
        u2 = (us*td  )
        u3 = (us*td-1)
        x  = ( 1*u0+0)*(0<=u0)*(u0<1) 
        x += ( 1*u1+1)*(0<=u1)*(u1<1)
        x += (-1*u2+2)*(0<=u2)*(u2<1) 
        x += (-1*u3+1)*(0<=u3)*(u3<1)
        x /= 4/us       
    elif kernelType=='Binomial5': # Binomial with Piecewise Linear Interpolation
        # This kernel is based on the binomial distribution
        # with piecewise linear interpolation. It is narrow, 
        # but still reproduces lines and constants
        
        #  0  1  4  6  4  1  0 <- Value
        # -3 -2 -1  0  1  2  3 <- Time
        us = 2 # Decrease the spacing by this amount
        
        u0 = (us*td+3)
        u1 = (us*td+2)
        u2 = (us*td+1)
        u3 = (us*td  )
        u4 = (us*td-1)
        u5 = (us*td-2)
        x  = ( 1*u0+0)*(0<=u0)*(u0<1) 
        x += ( 3*u1+1)*(0<=u1)*(u1<1)
        x += ( 2*u2+4)*(0<=u2)*(u2<1) 
        x += (-2*u3+6)*(0<=u3)*(u3<1)
        x += (-3*u4+4)*(0<=u4)*(u4<1)
        x += (-1*u5+1)*(0<=u5)*(u5<1)
        x /= (16)/us
    elif kernelType=='Exponential': # Binomial with Piecewise Linear Interpolation        
        alpha = 2 # Could be pretty much any positive value
        if not np.isnan(parameter):
            alpha = parameter
        x  = np.zeros_like(td)
        b1 = (0.0<=abs(td))&(abs(td)<0.5) 
        b2 = (0.5<=abs(td))&(abs(td)<1.0)
        x[b1] = (1-np.power(0.5,1-alpha)*np.power(       np.abs(td[b1])   ,alpha))
        x[b2] = (  np.power(0.5,1-alpha)*np.power(np.abs(np.abs(td[b2])-1),alpha))      
    elif kernelType=='Cubic Basis':
        a = -1.0
        ua = (td+2)
        ub = (td+1)
        uc = (td  )
        ud = (td-1)
        x  = (1/6)*(ua)**3                 *(0<=ua)*(ua<1) 
        x += (1/6)*(1+3*ub+3*ub**2-3*ub**3)*(0<=ub)*(ub<1)
        x += (1/6)*(4-6*uc**2+3*uc**3)     *(0<=uc)*(uc<1)
        x += (1/6)*(1-ud)**3               *(0<=ud)*(ud<1)
    elif kernelType=='Quadratic Interpolant':
        # This is an inefficient and lazy implementation, but mathematically
        # will produce the same result as a normal cubic spline
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
        # for a list of other types (kinds) of interpolation
        x = np.arange(-200,200)
        y = (x==0)
        function = scipy.interpolate.interp1d(x, y, kind='quadratic')
        x = function(td)        
    elif kernelType=='Cubic Interpolant':
        # This is an inefficient and lazy implementation, but mathematically
        # will produce the same result as a normal cubic spline
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
        # for a list of other types (kinds) of interpolation
        x = np.arange(-200,200)
        y = (x==0)
        function = scipy.interpolate.interp1d(x, y, kind='cubic')
        x = function(td)
    elif kernelType=='Cubic':
        a = -1
        x = (1-(a+3)*td**2 + (a+2)*np.abs(td)**3)*(np.abs(td)<1)
        x += a*(np.abs(td)-1)*(np.abs(td)-2)**2.0*(1<=np.abs(td))*(np.abs(td)<2)        
    elif kernelType=='Cubic (Quadratic Reproducing)':
        a = -0.5
        x = (1-(a+3)*td**2 + (a+2)*np.abs(td)**3)*(np.abs(td)<1)
        x += a*(np.abs(td)-1)*(np.abs(td)-2)**2.0*(1<=np.abs(td))*(np.abs(td)<2)
    else:
        print('Uknown kernel type "%s"' % kernelType)
        raise
    return(x)
def KernelInterpolate(image,ri,ci,kernelType='Binomial5'):
    # Determines the value of the image at the value ri,ci where
    # ri,ci do not need to be interger (pixel) values
    # This is useful for resampling images at new values that are
    # not necessarily on a grid
    kernelWidth = 2     # Kernel width in units of pixels, varies by kernel. Doesn't hurt if this is larger than it needs to be
    kernelType  = 'Binomial5'

    columnMin = max(int(np.floor(ci-kernelWidth)),0               )
    columnMax = min(int(np.ceil (ci+kernelWidth)),image.shape[1]-1)
    rowMin    = max(int(np.floor(ri-kernelWidth)),0               )
    rowMax    = min(int(np.ceil (ri+kernelWidth)),image.shape[0]-1) 

    zi = 0
    for irv,ic in enumerate(range(columnMin,columnMax+1)):
        z = 0
        for ir in range(rowMin,rowMax+1):
            z += image[ir,ic]*Kernel(ri,ir,kernelType=kernelType)
        zi += z*Kernel(ci,ic,kernelType=kernelType)
    return(zi)
