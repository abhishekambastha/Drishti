import numpy as np

def PCA(A):
	C = np.dot(A.T,A)
	[eVal, eVect] = np.linalg.eigh(C)
	eVect = np.dot(A,eVect)
	index = np.argsort(-eVal)
	eVect = eVect[:,index]
	return eVal, eVect