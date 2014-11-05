import os, sys
from PIL import Image
import numpy as np

def main():
	path = "/Users/abhi/projDrishti/trainFaces"
	modelImages(path)
	pass

############# Get Files Subdirectory Wise and separately! ##############
def modelImages(path):
	model = []
	modelName=[]
	for dirname, dirnames, filenames in os.walk(path):
		if dirnames == []:
			model.append(filenames)
			modelName.append(dirname)
	X = []
	Y = []
	for i in range(len(modelName)):
		#print modelName[i], "has ", len(model[i]), # "files namely ", model[i]
		temp = np.empty((0, 77760), dtype=np.uint8)
		print "", i
		X.append([])
		
		for names in model[i]:
			modelImageLocation = modelName[i] + '/' + names
			im = Image.open(modelImageLocation)
			im.convert("L")
			im = np.asarray(im,dtype=np.uint8)
			im = im.flatten()
			print "shape of im = ", im.shape ##concat here soemhow
			temp = np.vstack((temp, im))
			X[i].append(im)    ##want images instead of
		Y.append(temp)

	print "Number of models = ", len(X), " samples=" , len(X[1]) 

	

	i =1
	univX = []

	

	for imList in X:
		for im in imList:
			print "An Image Vector :", i, im.shape
			univX.append(im)
			i = i+1 
	
	print "univ is of ", univX[0].shape, len(univX)
	print "Y is ", len(Y), Y[0].shape

	imgDimension = Y[0].T.shape
	A = np.empty((imgDimension[0],0))
	print "null A ", A.shape
	for imgs in Y:
		print imgs.T.shape
		A = np.hstack((A, imgs.T))

	print "A", A.shape
	#A = np.asarray(univX).T  
	#print A
	#print A.shape

def model(path):

	subdir = [dirnames for dirname, dirnames, filenames in os.walk("/Users/abhi/projDrishti/trainFaces")]
	
	print filenames

	for folder in subdir[0]:
		print path+"/" + folder

	mdl = []
	for filenam in subdir:
		mdl.append(filenam)

	
	
if __name__ == '__main__':
	main()