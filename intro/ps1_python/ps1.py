import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread('./input/ps1-input0.png')
    img = img[:,:,1]
    show(img)

    edges = cv2.Canny(img,100,200)

    show(edges)

def show(pic):
    cv2.imshow('image',pic);
    cv2.waitKey(0);
    cv2.destroyAllWindows();

if __name__=='__main__':
    main()
