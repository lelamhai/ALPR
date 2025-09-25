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
    imgOrigin = cv2.imread("assets/imgs/18.png")
    gray_image = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2GRAY)
    bilateralFilter = cv2.bilateralFilter(gray_image, 9, 75, 75)
    _, thresholded_otsu = cv2.threshold(bilateralFilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edged = cv2.Canny(thresholded_otsu, 10, 200)

    # img_copy = imgOrigin.copy()
    # contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    # for i, c in enumerate(contours):
    #     peri   = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     if len(approx) == 4:                     
    #         x, y, w, h = cv2.boundingRect(approx)
    #         cv2.drawContours(img_copy, [approx], -1, (255, 0, 255), 3)  # xanh lá, độ dày 3
    #         cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)  # xanh dương




    cv2.imshow('gray_image', gray_image)
    cv2.imshow('bilateralFilter', bilateralFilter)
    cv2.imshow('thresholded_otsu', thresholded_otsu)
    cv2.imshow('Canny', edged)





    # img_rgb = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2RGB)
    # grayImage = GrayImage(imgOrigin)
    # gray_image = cv2.bilateralFilter(grayImage, 11, 17, 17)


    # imgGaussianBlur = cv2.GaussianBlur(grayImage, (5,5), 0)
    # imgThresh = cv2.adaptiveThreshold(imgGaussianBlur, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    # # _, global_thresh = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
    # edged = cv2.Canny(grayImage, 10, 200)

    # img_copy = imgOrigin.copy()
    # contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    # for i, c in enumerate(contours):
    #     peri   = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     if len(approx) == 4:                     
    #         x, y, w, h = cv2.boundingRect(approx)
    #         cv2.drawContours(img_copy, [approx], -1, (255, 0, 255), 3)  # xanh lá, độ dày 3
    #         cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)  # xanh dương






    # cv2.imshow('Image', img_rgb[..., ::-1])
    # cv2.imshow('gray_image', gray_image)
    # cv2.imshow('GaussianBlur', imgGaussianBlur)
    # cv2.imshow('imgThresh', imgThresh)
    # cv2.imshow('Canny', edged)

    # plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()