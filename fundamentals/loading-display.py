import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('logo.png')

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
