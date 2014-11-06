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
for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/trainFaces/p1"):
	a=1;

print "Face Database Location ",dirname


## Populate the List X with 2D image arrays
P1,y = [],[]
c=0
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	P1.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1

model1 = asColumnMatrix(P1)

for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/trainFaces/p2"):
	a=1;

print "Face Database Location ",dirname



## Populate the List X with 2D image arrays
P2,y = [],[]
c=0
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	P2.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1

model2 = asColumnMatrix(P2)

for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/trainFaces/p3"):
	a=1;

print "Face Database Location ",dirname


## Populate the List X with 2D image arrays
P3,y = [],[]
c=0
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	P3.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1

model3 = asColumnMatrix(P3)

for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/trainFaces/p4"):
	a=1;

print "Face Database Location ",dirname


## Populate the List X with 2D image arrays
P4,y = [],[]
c=0
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	P4.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1

model4 = asColumnMatrix(P4)

for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/trainFaces/p5"):
	a=1;

print "Face Database Location ",dirname


## Populate the List X with 2D image arrays
P5,y = [],[]
c=0
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	P5.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1

model5 = asColumnMatrix(P5)

for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/trainFaces/p6"):
	a=1;

print "Face Database Location ",dirname


## Populate the List X with 2D image arrays
P6,y = [],[]
c=0
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	P6.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1

model6 = asColumnMatrix(P6)

for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/trainFaces/p7"):
	a=1;

print "Face Database Location ",dirname


## Populate the List X with 2D image arrays
P7,y = [],[]
c=0
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	P7.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1
model7 = asColumnMatrix(P7)

for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/trainFaces/p8"):
	a=1;

print "Face Database Location ",dirname


## Populate the List X with 2D image arrays
P8,y = [],[]
c=0
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	P8.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1


model8 = asColumnMatrix(P8)

#query image
