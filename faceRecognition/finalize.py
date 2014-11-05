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

def main():
	path = "/Users/abhi/projDrishti/trainFaces"
	[A, modelList] = modelImages(path)
	[eVal, eVect] = PCA(A)

	meanAList = getParameters(modelList,eVect)

	print "In Main shape of parametersList is " ,  meanAList[0].shape
	pass


############# Get Files Subdirectory Wise and separately! ##############
def modelImages(path):

	modelFileName = [] 	#List of a List of File Names (one List per person)
	modelPath=[]  		#List of File Paths (one subdir per Person)

						##Get the file Locations
	for dirname, dirnames, filenames in os.walk(path):
		if dirnames == []:
			modelFileName.append(filenames)
			modelPath.append(dirname)
	
	modelList = []		#List of image vectors
	a = np.empty((0, imgDim), dtype=np.uint8)
	for i in range(len(modelPath)):
		#print modelPath[i], "has ", len(modelFileName[i]), # "files namely ", modelFileName[i]
		
		temp = np.empty((0, imgDim), dtype=np.uint8) # temp variable to store image vectors per model
	
		for filename in modelFileName[i]:
			modelImageLocation = modelPath[i] + '/' + filename
			im = Image.open(modelImageLocation)
			im.convert("L")
			im = np.asarray(im,dtype=np.uint8)
			im = im.flatten()    #row vector
			#print "shape of im = ", im.shape 
			temp = np.vstack((temp, im))   			##for individual models
			a = np.vstack((a, im))   				##for entire A vector
		
		modelList.append(temp.T)

	A = a.T

	print "A", A.shape

	for i in range(len(modelList)):
		print "Model ", i, " is of ", modelList[i].shape

	return A, modelList


def PCA(A):
	C = np.dot(A.T,A)
	print "Dimensions of C is ", C.shape
	[eVal, eVect] = np.linalg.eigh(C)
	eVect = np.dot(A,eVect)
	index = np.argsort(-eVal)
	eVect = eVect[:,index]

	print eVect.shape
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

	print "aList has ", len(eigenList), " elements of dimension ",eigenList[0].shape
	print "meanAList has ", len(meanEigenList), " elements of dimension ",meanEigenList[0].shape

	return meanEigenList


	
if __name__ == '__main__':
	main()