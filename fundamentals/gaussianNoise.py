import numpy as np
import cv2

img = cv2.imread('logo.png')
print "Image Dimensions", img.shape

cv2.imshow('image',img[:20,:,1])
cv2.waitKey(0)
cv2.destroyAllWindows()
gauss = np.random.normal(0,1)
