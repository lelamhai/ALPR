import cv2
import numpy as np
from PIL import Image, ImageFilter
import math
import matplotlib.pyplot as plt
from math import cos, pi

import os
out_dir = "outputs/plates"
os.makedirs(out_dir, exist_ok=True)


def GrayImage(imgOG):
    mgHue, imgSaturation, imgValue = cv2.split(imgOG)
    return imgValue

def main():
    imgOrigin = cv2.imread("assets/imgs/21.1.png")
    gray_image = GrayImage(imgOrigin)
    width, height = np.array(gray_image).shape
    imgBlurred = cv2.GaussianBlur(gray_image, (5,5), 0)
    bilateralFilter = cv2.bilateralFilter(imgBlurred, 9, 17, 17)
    _, thresholded_otsu = cv2.threshold(bilateralFilter, 127, 255, cv2.THRESH_BINARY)
    edge = cv2.Canny(thresholded_otsu, 10, 200)

    H, W = edge.shape[:2]

    fw = 100   # width px
    fh = 25    # height px

    vis = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    k = 0
    for y in range(0, H, fh):
        for x in range(0, W, fw):
            hue = (k * 35) % 180
            bgr = cv2.cvtColor(np.uint8([[[hue,255,255]]]), cv2.COLOR_HSV2BGR)[0,0]
            color = int(bgr[0]), int(bgr[1]), int(bgr[2])
            cv2.rectangle(vis, (x, y), (x+fw, y+fh), color, 2)
            k += 1

    cv2.imwrite("grid_custom.png", vis)
    cv2.imshow("gray_image",gray_image)
    cv2.imshow("bilateralFilter",bilateralFilter)
    cv2.imshow("thresholded",thresholded_otsu)
    cv2.imshow("edged",edge)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()