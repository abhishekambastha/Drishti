import numpy as np
import PIL.Image as Image
import os, sys

for dirname, dirnames, filenames in os.walk("/home/abhishek/faces"):
	print filenames

X,y = [],[]
c=1
for name in filenames:
	s = "" + dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	X.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1

for i in range (0, len(X)):
	print (X[i].T).shape
	print X[i].shape

