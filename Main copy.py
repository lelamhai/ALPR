import cv2
import numpy as np
from PIL import Image, ImageFilter
import math
import matplotlib.pyplot as plt

def GrayImage(imgOG):
    mgHue, imgSaturation, imgValue = cv2.split(imgOG)
    return imgValue

def Sobel(imgGray: Image.Image, width, height):
    Sx = np.array([[-1, -2, -1],[ 0, 0, 0],[ 1, 2, 1]])
    Sy = np.array([[ 1, 0, -1],[ 2, 0, -2],[ 1, 0, -1]])
    
    D0 = 127
    EdgeSobel = imgGray.convert()
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            XR = 0
            YR = 0
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    pixel_value = imgGray.getpixel((i, j))
                    XR += pixel_value * Sx[i - (x - 1), j - (y - 1)]
                    YR += pixel_value * Sy[i - (x - 1), j - (y - 1)]
            Mag = math.sqrt(XR * XR + YR * YR)
            if Mag <= D0:
                Mag = 0
            else:
                Mag = 255
            EdgeSobel.putpixel((x, y), Mag) 

    EdgeSobel = np.array(EdgeSobel)

    return EdgeSobel


def main():
    imgOrigin = cv2.imread("assets/imgs/12.jpg")
    gray_image = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2GRAY)
    bilateralFilter = cv2.bilateralFilter(gray_image, 9, 17, 17)
    _, thresholded_otsu = cv2.threshold(bilateralFilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edged = cv2.Canny(thresholded_otsu, 10, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(edged, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True) [:5]

    image2 = imgOrigin.copy()
    cv2.drawContours(image2,contours,-1,(0,255,0),3)
    cv2.imshow("edged",edged)
    cv2.imshow("dilated_image",dilated_image)
    cv2.imshow("Top 30 contours",image2)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()