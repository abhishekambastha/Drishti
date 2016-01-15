import numpy as np
import cv2

img = cv2.imread('logo.png')
imgB = img[:,:,1];
cv2.imshow('image',img);

#Adding Noise
s = img.shape
noise = np.random.randn(s[0],s[1],s[2])


sigma = int(raw_input());

imgX = img + noise*sigma

cv2.imshow('image',imgX);
cv2.waitKey(0)
cv2.destroyAllWindows()


