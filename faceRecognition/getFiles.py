 
import numpy as np 
import os, sys
from PIL import Image
import pca

 #arbit

##recurse till no subdir found and start reading files-> populate A matrix , and separate modelList matrix
def getImageList(path):

	numberOfModels=0
	imageResolution=77760
	modelList = []
	identityList = []
	newIdentityList = []
	A = np.empty((0, imageResolution), dtype=np.uint8)

	for dir, subdir, file in os.walk(path):
		if(subdir==[]):
			numberOfModels += 1
			tempA = np.empty((0, imageResolution), dtype=np.uint8)
			identityList.append(dir)
			for filename in file:
				if(filename[0] != '.'):
					imageURL = dir + "/" + filename
					im = getImageVector(imageURL)
					A = np.vstack((A, im))
					tempA = np.vstack((tempA, im))
			modelList.append(tempA.T)

	for names in identityList:
		newIdentityList.append(names[-2]+names[-1])

	return A.T, modelList, newIdentityList


def getImageVector(imageLocation):
	im = Image.open(imageLocation)
	im.convert("L")
	im = np.asarray(im,dtype=np.uint8)
	im = im.flatten()
	return im

##for testing and debugging to be removed later
def testImage(testLoc, model, eVect, identityList):
	for dirname, dirnames, filenames in os.walk(testLoc):
		a=0
	testImageList = []
	fName = []
	for name in filenames:
		s = dirname +"/" + name
		try:
			im = Image.open(s)
			im.convert("L")
			testImageList.append(np.asarray(im, dtype=np.uint8).ravel().T)
			fName.append(name)
		except IOError:
			print "Cannot Open ", name

	for i in range(len(testImageList)):
		R = pca.getModelName(np.asarray(testImageList[i]), model, eVect)
		print "Image" , fName[i], " identified as: ", identityList[R]



def main():
	path = "/Users/abhi/projDrishti/trainFaces"
	query = "/Users/abhi/projDrishti/testFaces"

	A, modelList,identityList = getImageList(path)
	eVal, eVect = pca.PCA(A)
	model = pca.generateModels(modelList,eVect)

	testImage(query, model, eVect, identityList)

	pass

if __name__ == '__main__':
	main()