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

print "Dimensions of single image", np.asarray(im).shape
print "Dimensions of X", np.asarray(X).shape


## Take the List of 2D images and return a List of long vectors u
def asRowMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((0, X[0].size), dtype=X[0].dtype)
	for row in X:
		mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
	return mat



## The Matrix Containing the rows of vectorised images, Y is actually A^T, refer notes!
Y = asRowMatrix(X)

print "Dimension of Y", Y.shape

## Compute eigenvalues and eigenvectors of the matrix C = X^T.X, Correlation Matrix 
def pca(X, y, num_components=0):
	[n,d] = X.shape    #number of rows n is the number of images = no. of ppl x sample of each
	if (num_components <= 0) or (num_components>n):
		num_components = n
	mu = X.mean(axis=0)  #sums up the rows
	X = X - mu
	if n>d:
		C = np.dot(X.T,X)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
	else:
		C = np.dot(X,X.T)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T,eigenvectors)
		for i in xrange(n):
			eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	# select only num_components
	eigenvalues = eigenvalues[0:num_components].copy()
	eigenvectors = eigenvectors[:,0:num_components].copy()
	return [eigenvalues, eigenvectors, mu]


[y, dim] = Y.shape
print y


[D, W, mu] = pca(Y,y)

print "eigenvalues ",D.shape
print "EigenVectors ", W.shape
print "mean :", mu.shape


def normalize(X, low, high, dtype=None):
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	# normalize to [0...1].	
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	X = X * (high-low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype=dtype)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create_font(fontname='Tahoma', fontsize=10):
	return { 'fontname': fontname, 'fontsize':fontsize }

def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
	fig = plt.figure()
	# main title
	fig.text(.5, .95, title, horizontalalignment='center') 
	for i in xrange(len(images)):
		ax0 = fig.add_subplot(rows,cols,(i+1))
		plt.setp(ax0.get_xticklabels(), visible=False)
		plt.setp(ax0.get_yticklabels(), visible=False)
		if len(sptitles) == len(images):
			plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',10))
		else:
			plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',10))
		plt.imshow(np.asarray(images[i]), cmap=colormap)
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename)
		
def imsave(image, title="", filename=None):
	plt.figure()
	plt.imshow(np.asarray(image))
	plt.title(title, create_font('Tahoma',10))
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename)


import matplotlib.cm as cm

# turn the first (at most) 16 eigenvectors into grayscale
# images (note: eigenvectors are stored by column!)
E = []
for i in xrange(min(len(X), 16)):
    e = W[:,i].reshape(X[0].shape)
    E.append(normalize(e,0,255))
# plot them and store the plot to "python_eigenfaces.pdf"
subplot(title="Eigenfaces AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet, filename="python_pca_eigenfaces.png")

def project(W, X, mu=None):
	if mu is None:
		return np.dot(X,W)
	return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
	if mu is None:
		return np.dot(Y,W.T)
	return np.dot(Y, W.T) + mu


 # reconstruction steps
steps=[i for i in xrange(10, min(len(X), 320), 20)]
E = []
for i in xrange(min(len(steps), 16)):
    numEvs = steps[i]
    P = project(W[:,0:numEvs], X[0].reshape(1,-1), mu)
    R = reconstruct(W[:,0:numEvs], P, mu)
    # reshape and append to plots
    R = R.reshape(X[0].shape)
    E.append(normalize(R,0,255))
   # plot them and store the plot to "python_reconstruction.pdf"
subplot(title="Reconstruction AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenvectors", sptitles=steps, colormap=cm.gray, filename="python_pca_reconstruction.png")
