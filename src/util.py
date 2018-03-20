import numpy as np
import cv2

def cnnProcessEyes(color, dim):
    colorDim = dim[:]
    colorDim.append(4)

    colorResize = cv2.resize(color, tuple(dim))
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    grayResize = cv2.resize(gray, tuple(dim))

    imsize = np.shape(color)
    
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               dp=16,
                               minDist=imsize[0],
                               minRadius=imsize[0]/10,
                               maxRadius=imsize[0]/2)

    print "circles: {}".format(circles)
    drawn = color[:] # keep a copy yo draw on
    alpha = np.zeros(np.shape(gray)) # single channel of same dim.
    if (circles is not None):
        print "circles {}".format(len(circles))
        for c in circles[0]:
            print "C: {}".format(c)
            cv2.circle(drawn, (c[0],c[1]), c[2], (0,255,0), 5)
            cv2.circle(alpha, (c[0],c[1]), c[2], 1, -1)

    drawnResize = cv2.resize(drawn, tuple(dim))
    alphaResize = cv2.resize(alpha, tuple(dim))

    fDim = colorDim[0:2]
    fDim.append(4) # Use 4 channel out with alpha channel with pupil
    # scale everything to 0->1
    print "colorDim: {}".format(colorDim)
    features = np.empty(colorDim)
    print np.shape(features)
    features[:,:,0:3] = colorResize*1.0/255
    features[:,:,3] = alphaResize*1.0/255
    
    return (features,drawnResize)

def reshapeToBatch(a):
    dim = [1]
    dim.extend(np.shape(a))
    b = np.reshape(a,dim)
    print np.shape(b)
    return b
