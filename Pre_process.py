import cv2 as cv
import numpy as np
import Detected_plate
gaussian_flt = (5,5)
kernel = np.ones((5,5))
def Show(name,img):
    cv.imshow(name, img)
    cv.waitKey()
# make gray scale image
#blur the image
#get the max contrast image
# make thresh image
def ImageThresh(imgOriginal):
    img_gray = getImgGray(imgOriginal)
    # cv.imshow("ImgGray",img_gray)
    tophat = cv.morphologyEx(img_gray, cv.MORPH_TOPHAT, kernel) # use tophat
    blackhat = cv.morphologyEx(img_gray, cv.MORPH_BLACKHAT, kernel) #use blackhat

    imgGrayPlusTophat = cv.add(img_gray,tophat)                 # add tophat and minus blackhat in order to take get the max

    imgMaxContrast = cv.subtract(imgGrayPlusTophat,blackhat)    #contrast of the image
    # cv.imshow("imgMaxContrast",imgMaxContrast)
    img_blur = cv.GaussianBlur(imgMaxContrast,gaussian_flt,0) # blur the image
    # cv.imshow("ImgBlur", img_blur)
    img_thresh = cv.adaptiveThreshold(img_blur,255.0,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,21,10)
    # cv.imshow("ImgThresh", img_thresh)
    cv.waitKey()
    return imgMaxContrast,img_thresh
def getImgGray(img):
    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    imgH, imgS, imgV = cv.split(img_HSV)
    return imgV
