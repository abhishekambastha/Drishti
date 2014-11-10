 
import numpy as np 
import os, sys
from PIL import Image
import pca

 #arbit

##recurse till no subdir found and start reading files
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

def populateModel(modelList, eVect):
	
	modelParameters = []
	
	#confusion in maths here
	for i in range(len(modelList)):
		modelParameters.append(np.dot(modelList[i].T, eVect))

	print len(modelParameters)
	print modelParameters[0].shape
	return modelParameters  #return coefficients obtained!





def getImageVector(imageLocation):
	im = Image.open(imageLocation)
	im.convert("L")
	im = np.asarray(im,dtype=np.uint8)
	im = im.flatten()
	return im

def main():
	path = "/Users/abhi/projDrishti/trainFaces"
	A, modelList,identityList = getImageList(path)
	eVal, eVect = pca.PCA(A)
	populateModel(modelList,eVect)
	pass

if __name__ == '__main__':
	main()