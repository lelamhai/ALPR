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

    for i in range(0, w - 32, 4):   # bước vẫn là 4 pixel
        for j in range(0, h - 16, 4):
            # ROI duy nhất: 32x16
            roi = edge[j:j+16, i:i+32]

            nonZero = cv2.countNonZero(roi)

            # Ngưỡng: bạn có thể chỉnh. 
            # Ví dụ: 15*4 = 60 vì hồi trước 4 ô mỗi ô >15 pixel
            if nonZero > 50:  
                dst[j:j+16, i:i+32] = roi


    
    # cv2.imshow("gray_image",gray_image)
    # cv2.imshow("bilateralFilter",bilateralFilter)
    # cv2.imshow("thresholded",thresholded_otsu)
    cv2.imshow("edged",edge)
    cv2.imshow("dst",dst)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()