import numpy as np

def PCA(A):
	C = np.dot(A.T,A)
	[eVal, eVect] = np.linalg.eigh(C)
	eVect = np.dot(A,eVect)
	index = np.argsort(-eVal)
	eVect = eVect[:,index]
	return eVal, eVect

def generateModels(modelList, eVect):
	
	modelParameters = []
	
	#confusion in maths here

	print "evect", eVect.shape
	print "imageList", modelList[0].T.shape
	for i in range(len(modelList)):
		modelParameters.append(np.dot(modelList[i].T, eVect))

	#print len(modelParameters)
	#print modelParameters[0].shape   # contains a for an image in a row, corresponding to 0 model
	return modelParameters  #return coefficients obtained!

def getModelName(u, modelList, eVect):

	meanModel = []
	for i in range(len(modelList)):
		meanModel.append(modelList[i].mean(axis=0))

	params = np.dot(u.T, eVect)

	dist = []
	for i in range (len(meanModel)):
		a1 = meanModel[i]
		a2 = params
		d = np.linalg.norm(a1-a2)
		dist.append(d)

	distArray = np.asarray(dist)
	res = distArray.argmin()
	return res



