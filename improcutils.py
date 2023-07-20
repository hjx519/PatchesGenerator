from scipy.signal import convolve2d as conv2d
import skimage
import numpy as np

def makegauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def drawpolyintoemptycanvas(CS,x,y,tx,ty):
    """
    Purpose is to draw a polygon into an empty canvas (image)
    for the purpose of subseuqntly combining it with an existing canvas
    The offsets can be chosen so that the polygon does not exceed
    the boundaries of the polygon
    """
    from skimage.draw import polygon

    img = np.zeros(CS, dtype=float)

    R = CS[1]-(ty+y)
    C = (tx+x)

    rr, cc = polygon(R, C)
    
    insiderlocs = (rr>=0) & (rr<CS[1])
    insideclocs = (cc>=0) & (cc<CS[0])
    insidelocs = insiderlocs & insideclocs
    
    img[rr[insidelocs],cc[insidelocs]] = 1
    
    return img


def addnoise(x,swn=0.1,ssn=0.1,gks=0.5,rectify=False):
    """
    Adds some white and structured noise; wn is non-Gaussian
    Usage: addnoise(x,swn,ssn,rectify)
    swn: white noise variance (it is zero mean by default)
    ssn: coloured noise, filtered by a 3x3 Gaussian kernel, var:0.25 (default)
         kernel size is scaled according to spatial sd of Gaussian kernel
    gks: Gaussian kernel sigma for the correlated noise
    rectify: Boolean - says whether or not image is rectified to remove negative values
    """
    m,n = x.shape
    wn = swn*np.random.randn(m,n)
    gksize = (6*gks,6*gks)
    g2 = makegauss2D(shape=gksize,sigma=gks)
    sn = conv2d(ssn*np.random.randn(m,n),g2,'same')
    
    y = x + wn + sn
    
    if rectify:
        y = y*(y>0)
    
    return(y)