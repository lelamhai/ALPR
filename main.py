import cv2
import numpy as np
from PIL import Image, ImageFilter
import math

def Lightness(imgOG: Image.Image, width, height):
    imgLightness = Image.new("L", (width, height))
    for x in range(width):
        for y in range(height):
            r, g, b = imgOG.getpixel((x, y))
            maxgray = max(r, g, b)
            mingray = min(r, g, b)
            Gray = (maxgray + mingray) // 2
            imgLightness.putpixel((x, y), Gray)
    return np.array(imgLightness)


def Sobel(imgGray: Image.Image, width, height):
    Sx = np.array([[-1, -2, -1],[ 0, 0, 0],[ 1, 2, 1]])
    Sy = np.array([[ 1, 0, -1],[ 2, 0, -2],[ 1, 0, -1]])
    
    D0 = 200
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
    PATH = 'assets/imgs/17.png'
    imgOG = Image.open(PATH)
    width, height = imgOG.size
    imgLightness = Lightness(imgOG, width, height)
    imgHistogram = cv2.equalizeHist(imgLightness)
    imgGaussianBlur = cv2.GaussianBlur(imgHistogram, (5,5), 0)
    imgSobel = Sobel(Image.fromarray(imgGaussianBlur), width, height)
       

    # cv2.imshow('Image', imgRGB)
    cv2.imshow('Lightness', imgLightness)
    cv2.imshow('Histogram', imgHistogram)
    cv2.imshow('GaussianBlur', imgGaussianBlur)
    cv2.imshow('Sobel', imgSobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()