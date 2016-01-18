import cv2
import numpy as np
from math import *

def main():
    img_edges = cv2.imread('./output/edges.png')
    img_edges = img_edges[:,:,1]
    acc = houghTransform(img_edges)
    show(acc)
    compare = cv2.HoughLines(img_edges,1,np.pi/180,200)
    print "End"



##Design Accumulator _theta_ = [0, 90]
##

def houghTransform(img_edges):
    rSize,cSize = img_edges.shape

    #accumulator array , rows -> d, cols-> theta
    accumulator = np.zeros((int(sqrt(rSize**2 + cSize**2)),180))

    for i in xrange(rSize):
        for j in xrange(cSize):
            if img_edges[i,j] == 255:
                for theta in xrange(180):
                    d = i*cos(radians(theta)) + j*sin(radians(theta))
    return accumulator

def show(pic):
    cv2.imshow('image',pic);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
if __name__=='__main__':
    main()
