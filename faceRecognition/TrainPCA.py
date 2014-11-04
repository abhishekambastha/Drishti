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
c=1
for name in filenames:
	s = dirname +"/" + name
	im = Image.open(s)
	im.convert("L")
	X.append(np.asarray(im, dtype=np.uint8))
	y.append(c)
	c = c+1


## Take the List of 2D images and return a List of long vectors u
def asRowMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((0, X[0].size), dtype=X[0].dtype)
	for row in X:
		mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
	return mat



## The Matrix Containing the rows of vectorised images, Y is actually A^T, refer notes!

##confusing .... use A instead ! 
Y = asRowMatrix(X)

#print "Dimension of Y", Y.shape

## Compute eigenvalues and eigenvectors of the matrix C = X^T.X, Correlation Matrix 
def pca(X, y, num_components=0):
	[n,d] = X.shape    #number of rows n is the number of images = no. of ppl x sample of each
	if (num_components <= 0) or (num_components>n):
		num_components = n
	mu = X.mean(axis=0)  #sums up the rows / (number of rows)
	X = X - mu
	if n>d:
		C = np.dot(X.T,X)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
	else:
		C = np.dot(X,X.T)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)    #dim of eign vect is (no of ppl)x n
		eigenvectors = np.dot(X.T,eigenvectors)				#now dim is (resolution of image)xn
		for i in xrange(n):
			eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	# select only num_components
	eigenvalues = eigenvalues[0:num_components].copy()
	eigenvectors = eigenvectors[:,0:num_components].copy()
	return [eigenvalues, eigenvectors, mu]
