import cv2
import numpy as np
from PIL import Image, ImageFilter
import math

def Lightness(imgOG: Image.Image, width, height):
    imgLightness = imgOG.convert()
    for x in range(0, width):
        for y in range(0, height):
            r, g, b = imgOG.getpixel((x, y))
            maxgray = max(r, g, b)
            mingray = min(r, g, b)
            Gray = (int)((maxgray + mingray) / 2)
            imgLightness.putpixel((x, y), (Gray, Gray, Gray))

    return imgLightness

def Sobel(imgGray: np.ndarray, width, height):
    imgLightness = imgGray.convert()

    Sx = np.array([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]])
    Sy = np.array([[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]])

    D0 = 255/2
    EdgeSobel = imgLightness.convert()
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            XR = 0
            YR = 0
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    r, g, b = imgLightness.getpixel((i, j))
                    XR += r * Sx[i - (x - 1), j - (y - 1)]
                    YR += r * Sy[i - (x - 1), j - (y - 1)]
            Mag = math.sqrt(XR * XR + YR * YR)
            if Mag <= D0:
                Mag = 0
            else:
                Mag = 255
            EdgeSobel.putpixel((x, y), (Mag, Mag, Mag))

    EdgeSobel = np.array(EdgeSobel)

    return EdgeSobel

# def FindConturs(imgOG: Image.Image, imgSobel: Image.Image):
#     cnts,_ = cv2.findContours(imgSobel.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
#     for c in cnts:
#         perimeter = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
#         if len(approx) == 4:
#                 screenCnt = approx

#         x,y,w,h = cv2.boundingRect(c)
#         new_img=imgOG[y:y+h,x:x+w]
#         cv2.imwrite('./'+str(i)+'.png',new_img)
#         i+=1

def main():
    PATH = 'assets/imgs/17.png'
    imgOG = Image.open(PATH)
    width, height = imgOG.size
    imgLightness = Lightness(imgOG, width, height)


    gray_image = cv2.GaussianBlur(np.array(imgLightness), (5,5), 0)
    
    imgSobel = Sobel(imgLightness, width, height)
    edged = cv2.cvtColor(np.array(imgSobel), cv2.COLOR_BGR2GRAY)

    # imgEdgedCanny = cv2.Canny(gray_image, 10, 200)

    imgOG = np.array(imgOG)
    imgRGB = cv2.cvtColor(imgOG, cv2.COLOR_RGB2BGR)

    cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
    
    i = 0
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
                x,y,w,h = cv2.boundingRect(approx)
                new_img=imgRGB[y:y+h,x:x+w]
                new_img_np = np.array(new_img)
                cv2.imwrite('./'+str(i)+'.png',new_img_np)
                i+=1

       

    cv2.imshow('Image', imgRGB)
    cv2.imshow('Lightness', np.array(imgLightness))
    cv2.imshow('Sobel', imgSobel)
    # cv2.imshow('Canny', imgEdgedCanny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()