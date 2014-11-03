import numpy as np
import cv2
import os, sys
import PIL.Image as Image

for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/faces"):
	print filenames

print dirname
X,y = [],[]
c=1
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	X.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1

for i in range (0, len(X)):
	print (X[i].T).shape
	print X[i].shape

