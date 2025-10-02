import cv2
import numpy as np
from PIL import Image, ImageFilter
import math
import matplotlib.pyplot as plt

import os
out_dir = "outputs/plates"
os.makedirs(out_dir, exist_ok=True)

PIEXL = 15

def GrayImage(imgOG):
    mgHue, imgSaturation, imgValue = cv2.split(imgOG)
    return imgValue

def main():
    imgOrigin = cv2.imread("assets/imgs/2.jpg")
    gray_image = GrayImage(imgOrigin)
    width, height = np.array(gray_image).shape
    imgBlurred = cv2.GaussianBlur(gray_image, (5,5), 0)
    bilateralFilter = cv2.bilateralFilter(imgBlurred, 9, 17, 17)
    _, thresholded_otsu = cv2.threshold(bilateralFilter, 127, 255, cv2.THRESH_BINARY)
    edge = cv2.Canny(thresholded_otsu, 10, 200)

    h, w = edge.shape[:2]

    dst = np.zeros_like(edge)

    block_w, block_h = 128, 64

    for i in range(0, w, block_w):
        for j in range(0, h, block_h):
            roi = edge[j:j+block_h, i:i+block_w]
            nonZero = cv2.countNonZero(roi)

            if nonZero > 300:
                dst[j:j+block_h, i:i+block_w] = roi


    
    # cv2.imshow("gray_image",gray_image)
    # cv2.imshow("bilateralFilter",bilateralFilter)
    # cv2.imshow("thresholded",thresholded_otsu)
    cv2.imshow("edged",edge)
    cv2.imshow("dst",dst)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()