import os, sys
from PIL import Image
import numpy as np

## Let The Dimension of Image be M*N = imgDim
## Let U be the vectorized image (coloumn vector)
## Let A be [U1 U2 U2 .. ..... ..... Upn]  --horizontal stack of image coloumn vectors
## Let number of sample be totalSamples = Pn
## Let number of persons be P
## Let model[i] be the list of model arrays i.e [U1 U2 U3 ... Un]
## modelList is a list of long image vectors

imgDim = 77760

fName =[]
modName=[]


def main():
	
	path = "/Users/abhi/projDrishti/trainFaces"
	queryLocation = "/Users/abhi/projDrishti/yalefaces/subject02.centerlight"
	q = "/Users/abhi/projDrishti/testFaces"
	
	#results(path, queryLocation)
	imL = testImage(q)

	[A, modelList] = modelImages(path)
	[eVal, eVect] = PCA(A)

	meanAList = getParameters(modelList,eVect)


	for i in range(len(imL)):
		R = results(path, np.asarray(imL[i]), meanAList, eVect)
		print "Image" , fName[i], " identified as: ", (modName[R])[-2] + (modName[R])[-1]
	pass

def results(path, u, meanAList, eVect):

	
	params = np.dot(u.T, eVect)

	dist = []
	for i in range (len(meanAList)):
		a1 = meanAList[i]
		a2 = params
		d = np.linalg.norm(a1-a2)
		dist.append(d)

	distArray = np.asarray(dist)
	res = distArray.argmin()
	minimum = min(distArray)
	maximum = max(distArray)

	return res

def testImage(testLoc):
	
	for dirname, dirnames, filenames in os.walk(testLoc):
		a=0

	testImageList = []

	for name in filenames:
		
		s = dirname +"/" + name
		try:
			im = Image.open(s)
			im.convert("L")
			testImageList.append(np.asarray(im, dtype=np.uint8).ravel().T)
			fName.append(name)
		except IOError:
			print "Cannot Open ", name

	for names in testImageList:
		print "Test Image ", a, " is ", names.shape
		a = a+1
	
	return testImageList


def getImageVector(queryLocation):
	im = Image.open(queryLocation)
	im.convert("L")
	im = np.asarray(im,dtype=np.uint8)
	im = im.flatten()    #row vector
	return im.T

############# Get Files Subdirectory Wise and separately! ##############
def modelImages(path):

	modelFileName = [] 	#List of a List of File Names (one List per person)
	modelPath=[]  		#List of File Paths (one subdir per Person)

						##Get the file Locations
	for dirname, dirnames, filenames in os.walk(path):
		if dirnames == []:
			modelFileName.append(filenames)
			modelPath.append(dirname)
			modName.append(dirname)
	
	modelList = []		#List of image vectors
	a = np.empty((0, imgDim), dtype=np.uint8)
	for i in range(len(modelPath)):
		#print modelPath[i], "has ", len(modelFileName[i]), # "files namely ", modelFileName[i]
		
		temp = np.empty((0, imgDim), dtype=np.uint8) # temp variable to store image vectors per model
	
		for filename in modelFileName[i]:
			modelImageLocation = modelPath[i] + '/' + filename
			try:
				im = Image.open(modelImageLocation)
				im.convert("L")
				im = np.asarray(im,dtype=np.uint8)
				im = im.flatten()    #row vector
				#print "shape of im = ", im.shape 
				temp = np.vstack((temp, im))   			##for individual models
				a = np.vstack((a, im))   				##for entire A vector
			except IOError as e:
				print "Error: File Skipped"
		
		modelList.append(temp.T)

	A = a.T

	print "A", A.shape

	for i in range(len(modelList)):
		print "Model ", i, " is of ", modelList[i].shape

	return A, modelList


def PCA(A):
	C = np.dot(A.T,A)
	[eVal, eVect] = np.linalg.eigh(C)
	eVect = np.dot(A,eVect)
	index = np.argsort(-eVal)
	eVect = eVect[:,index]

	return eVal, eVect

def getParameters(modelList,eVect):

	eigenList =[]
	meanEigenList = []
	temp = 0;

	for p in modelList:
		temp = np.dot(p.T, eVect)
		eigenList.append(temp)

	for temp in eigenList:
		temp = temp.mean(axis=0)
		meanEigenList.append(temp)

	return meanEigenList


	
if __name__ == '__main__':
	main()