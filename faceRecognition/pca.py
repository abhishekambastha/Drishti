import numpy as np
import cv2
import os, sys
import PIL.Image as Image


##Edit Path as required !
for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/faces"):
	a=1;

print "Face Database Location ",dirname


## Populate the List X with 2D image arrays
X,y = [],[]
c=0
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	X.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1

def asColumnMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
	for col in X:
		mat = np.hstack((mat, np.asarray(col).reshape(-1,1)))
	return mat

print "check A[0].size", X[0].size
A = asColumnMatrix(X)

print np.asarray(im).shape
print A.shape

C = np.dot(A.T, A)

print C.shape

[eVal, eVect] = np.linalg.eigh(C)

print eVect.shape

eVect = np.dot(A,eVect)

print eVect.shape
index = np.argsort(-eVal)
eVect = eVect[:,index]

print eVect.shape

image = np.asarray(im).reshape(-1,1)

print image.shape

x = np.dot(A.T,eVect)

print x.shape

#create model for person 1,2,3......................

#query image
