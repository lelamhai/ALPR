import cv2
import numpy as np
from PIL import Image, ImageFilter
import math
import matplotlib.pyplot as plt
import os
out_dir = "outputs/plates"
os.makedirs(out_dir, exist_ok=True)

PIEXL = 15

def LoadImageOG(path): 
    imgOG = cv2.imread(path) 
    cv2.imshow("LoadImageOG", imgOG)
    return imgOG
    

def GrayImage(imgOG):
    mgHue, imgSaturation, imgValue = cv2.split(imgOG)
    # cv2.imshow("GrayImage", imgValue)
    return imgValue

def Binarization(src):
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


    # cv2.imshow("Binarization", dst)
    return dst

   
    
def Morphology(imgBinarization):
    S1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # kernel 3x1
    img = cv2.dilate(imgBinarization, None, iterations=12)
    img = cv2.erode(img, None, iterations=12)
    img = cv2.dilate(img, S1, iterations=19)
    img = cv2.erode(img, S1, iterations=20)
    img = cv2.dilate(img, None, iterations=11)
    cv2.imshow("Morphology", img)

    return img


def ShowPlate(imgOG, imgProcessing):
    width, height = np.array(imgProcessing).shape

    # --- CHUYỂN ẢNH SANG MÀU ---
    imgProcessing = cv2.cvtColor(imgProcessing, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(
        cv2.cvtColor(imgProcessing, cv2.COLOR_BGR2GRAY),
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    i = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if w >= width or h >= height:
            continue

        radio = (width * height) / area

        # --- VẼ CONTOUR ---
        cv2.drawContours(imgProcessing, [c], -1, (0, 255, 255), 2)

        # --- GHI SỐ i MÀU ĐỎ, CÓ VIỀN ĐEN ---
        cv2.putText(imgProcessing, str(i), (x, y - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(imgProcessing, str(i), (x, y - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

        i += 1
        cv2.imshow("FinderPlate", imgProcessing)



def FinderPlate(imgOG, imgProcessing):
    width, height = np.array(imgProcessing).shape
    contours, hierarchy = cv2.findContours(imgProcessing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    crops = []
    i=0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True) 
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        ar = w / float(h)
      
        if w > width or h > height:
            continue
        
        radio =  (width*height)/area

        print(f"{i}: {radio} : {ar:.2f}") 
        i+=1
        if(radio > 30 and radio <270):
            if(ar<2.6 and ar<7):
                crop = imgOG[y:y+h, x:x+w].copy()
                crops.append(crop)
                cv2.imwrite(os.path.join(out_dir, f"plate_{len(crops):02d}.png"), crop)




def main():
    PATH_IMAGE = "assets/imgs/1.png"
    imgOrigin = LoadImageOG(PATH_IMAGE)
    imgGrayscale = GrayImage(imgOrigin)
    imgBinarization = Binarization(imgGrayscale)
    imgMorphology = Morphology(imgBinarization)
    ShowPlate(imgOrigin, imgMorphology)
    FinderPlate(imgOrigin, imgMorphology)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()