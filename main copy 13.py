import cv2
import numpy as np
from PIL import Image, ImageFilter
import math
import matplotlib.pyplot as plt

PIEXL = 15
def GrayImage(imgOG):
    mgHue, imgSaturation, imgValue = cv2.split(imgOG)
    return imgValue

def Processing(src):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
    bothat = cv2.normalize(blackhat, blackhat, 0, 255, cv2.NORM_MINMAX)

    mean_val = cv2.mean(bothat)[0]
    thresh_val = int(10 * mean_val)
    _, thresholded = cv2.threshold(bothat, thresh_val, 255, cv2.THRESH_BINARY)

    h, w = thresholded.shape[:2]
    dst = np.zeros_like(thresholded)

    for i in range(0, w - 32, 4):
        for j in range(0, h - 16, 4):
            # 4 ROI 16x8
            roi1 = thresholded[j:j+8, i:i+16]
            roi2 = thresholded[j:j+8, i+16:i+32]
            roi3 = thresholded[j+8:j+16, i:i+16]
            roi4 = thresholded[j+8:j+16, i+16:i+32]

            nonZero1 = cv2.countNonZero(roi1)
            nonZero2 = cv2.countNonZero(roi2)
            nonZero3 = cv2.countNonZero(roi3)
            nonZero4 = cv2.countNonZero(roi4)

            # Đếm số ô vượt ngưỡng 15
            cnt = 0
            if nonZero1 > PIEXL: cnt += 1
            if nonZero2 > PIEXL: cnt += 1
            if nonZero3 > PIEXL: cnt += 1
            if nonZero4 > PIEXL: cnt += 1

            # Nếu > 2 ô "đậm", copy block 32x16 từ mini_thresh sang dst
            if cnt > 2:
                dst[j:j+16, i:i+32] = thresholded[j:j+16, i:i+32]



    S1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # kernel 3x1

    dst1 = cv2.dilate(dst, None, iterations=12)
    dst1 = cv2.erode(dst1, None, iterations=12)
    dst1 = cv2.dilate(dst1, S1, iterations=19)
    dst1 = cv2.erode(dst1, S1, iterations=20)
    dst1 = cv2.dilate(dst1, None, iterations=11)


    cv2.imshow("Original", src)
    cv2.imshow("thresholded", thresholded)
    cv2.imshow("dst", dst)
    cv2.imshow("dst1", dst1)

def FinderPlate(imgThresh):

    return


def main():
    imgOrigin = cv2.imread("assets/imgs/17.png")
    gray_image = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2GRAY)
    imgGrayscale = GrayImage(imgOrigin)
    Processing(imgGrayscale)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()