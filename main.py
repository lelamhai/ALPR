import cv2
import numpy as np
from PIL import Image, ImageFilter
import math
import matplotlib.pyplot as plt
import easyocr
import os
out_dir = "data/chars"
os.makedirs(out_dir, exist_ok=True)

PIEXL = 15

def LoadKNN():
    npaClassifications = np.loadtxt("assets/classifications.txt", np.float32)
    npaFlattenedImages = np.loadtxt("assets/flattened_images.txt", np.float32)
    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
    kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

def LoadImageOG(path): 
    imgOG = cv2.imread(path) 
    # cv2.imshow("LoadImageOG", imgOG)
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
    
    #cv2.imshow("Binarization", thresholded)
    return thresholded


def FilterNoise(src):
    h, w = src.shape[:2]
    dst = np.zeros_like(src)

    for i in range(0, w - 32, 4):
        for j in range(0, h - 16, 4):
            # 4 ROI 16x8
            roi1 = src[j:j+8, i:i+16]
            roi2 = src[j:j+8, i+16:i+32]
            roi3 = src[j+8:j+16, i:i+16]
            roi4 = src[j+8:j+16, i+16:i+32]

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
                dst[j:j+16, i:i+32] = src[j:j+16, i:i+32]

    # cv2.imshow("Filter Noise", dst)

    return dst
   
    
def Morphology(imgBinarization):
    S1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # kernel 3x1
    img = cv2.dilate(imgBinarization, None, iterations=12)
    img = cv2.erode(img, None, iterations=12)
    img = cv2.dilate(img, S1, iterations=19)
    img = cv2.erode(img, S1, iterations=20)
    img = cv2.dilate(img, None, iterations=11)

    # cv2.imshow("Morphology", img)
    return img


def FinderPlates(imgProcessing):
    width, height = np.array(imgProcessing).shape
    contours, hierarchy = cv2.findContours(imgProcessing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = w / float(h)
      
        if w > width or h > height:
            continue
        
        radio =  (width*height)/area

        if 30 < radio < 270:
            if 1.2 < ar < 6:
                if 100 < w < 550 and 40 < h < 250:
                    boxes.append((x, y, w, h))
    return boxes

def SwapY(chars):
    n = len(chars)
    chars = list(chars)
    for i in range(n - 1):
        for j in range(0, n - 1 - i):
            if chars[j][1] > chars[j + 1][1]:
                chars[j], chars[j + 1] = chars[j + 1], chars[j]

    return chars


def SwapX(chars):
    n = len(chars)
    chars = list(chars)
    for i in range(n - 1):
        for j in range(0, n - 1 - i):
            if chars[j][0] > chars[j + 1][0]:
                chars[j], chars[j + 1] = chars[j + 1], chars[j]

    return chars
                


def DetectPlate(boxes, imgOG, imgBinarization):
    Plate = []
    for i, (x, y, w, h) in enumerate(boxes):
        img_crop = imgOG[y:y + h, x:x + w]
        img_Binarization = imgBinarization[y:y + h, x:x + w]
        imgGrayscale = GrayImage(img_crop)
        imgGrayscale = cv2.resize(imgGrayscale, (408, 70))
        img_Binarization = cv2.resize(img_Binarization, (408, 70))
        imgThresh = cv2.adaptiveThreshold(imgGrayscale, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
        contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for j, c in enumerate(contours):
            c_x, c_y, c_w, c_h = cv2.boundingRect(c)
            if 15<c_w<50  and 25<c_h<65:# and c_w*c_h>470:
                Plate.append((c_x, c_y, c_w, c_h))
        break 

    return Plate, imgGrayscale, img_Binarization

def drawboxes(chars, imgCrop, imgBinarization):
    imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_GRAY2BGR)
    listChars = []
    for i, (x, y, w, h) in enumerate(chars):
        char = imgBinarization[y:y+h, x:x+w].copy()
        listChars.append(char)
        cv2.imwrite(os.path.join(out_dir, f"char{i}.png"), char)
        cv2.rectangle(imgCrop, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("Contours Boxes", imgCrop)
    return listChars
 

def DetectChars(listChars):
    RESIZED_IMAGE_WIDTH = 20
    RESIZED_IMAGE_HEIGHT = 30
    npaClassifications = np.loadtxt("assets/KNN/classifications.txt", np.float32)
    npaFlattenedImages = np.loadtxt("assets/KNN/flattened_images.txt", np.float32)
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
    kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    for char in listChars:
        imgROIResized = cv2.resize(char, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize image
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)  # call KNN function find_nearest;
        strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of characters
        print(strCurrentChar)


def main():
    PATH_IMAGE = "assets/imgs/3.png"
    imgOrigin = LoadImageOG(PATH_IMAGE)
    imgGrayscale = GrayImage(imgOrigin)
    imgBinarization = Binarization(imgGrayscale)
    imgFileNoise = FilterNoise(imgBinarization)
    imgMorphology = Morphology(imgFileNoise)
    boxesPlate = FinderPlates(imgMorphology)
    plateChars, imgCrop, img_Binarization = DetectPlate(boxesPlate, imgOrigin, imgBinarization)
    plateChars = SwapX(plateChars)
    listChars = drawboxes(plateChars, imgCrop, img_Binarization)
    DetectChars(listChars)

    

if __name__ == "__main__":
    main()
    cv2.waitKey(0)