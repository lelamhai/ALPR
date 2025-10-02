import cv2
import numpy as np
from PIL import Image, ImageFilter
import math
import matplotlib.pyplot as plt

import os
out_dir = "outputs/plates"
os.makedirs(out_dir, exist_ok=True)

PIEXL = 15

def Processing(src):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
    bothat = cv2.normalize(blackhat, blackhat, 0, 255, cv2.NORM_MINMAX)

    mean_val = cv2.mean(bothat)[0]
    thresh_val = int(10 * mean_val)

    _, thresholded = cv2.threshold(bothat, thresh_val, 255, cv2.THRESH_BINARY)



    cv2.imshow("Original", src)
    cv2.imshow("BlackHat", blackhat)
    cv2.imshow("bothat", bothat)
    cv2.imshow("thresholded", thresholded)

def GrayImage(imgOG):
    mgHue, imgSaturation, imgValue = cv2.split(imgOG)
    return imgValue

def main():
    imgOrigin = cv2.imread("assets/imgs/1.jpg")
    imgGrayscale = GrayImage(imgOrigin)
    Processing(imgGrayscale)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()