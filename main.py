import cv2
import numpy as np
from PIL import Image, ImageFilter
import math

# Mở ảnh và resize
HinhGocPIL = Image.open('data/new/1.jpg')
imgHinhGocPIL = HinhGocPIL.resize((500, 350))
Width, Height = imgHinhGocPIL.size

##########################################
#               TIỀN XỬ LÍ ẢNH           #
##########################################

def find_max(a, b, c):
    max = a
    if max < b: max = b
    if max < c: max = c
    return max

def find_min(a, b, c):
    min = a
    if min > b: min = b
    if min > c: min = c
    return min

# Chuyển sang mức xám theo phương pháp Lightness
Lightness = imgHinhGocPIL.convert()
for x in range(0, Width):
    for y in range(0, Height):
        r, g, b = imgHinhGocPIL.getpixel((x, y))
        maxgray = find_max(r, g, b)
        mingray = find_min(r, g, b)
        Gray = (int)((maxgray + mingray) / 2)
        Lightness.putpixel((x, y), (Gray, Gray, Gray))

imgAnhGray = np.array(Lightness)



Sx = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]])

Sy = np.array([[ 1,  0, -1],
               [ 2,  0, -2],
               [ 1,  0, -1]])

D0 = 200
EdgeSobel = Lightness.convert()
for x in range(1, Width - 1):
    for y in range(1, Height - 1):
        XR = 0
        YR = 0
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                r, g, b = Lightness.getpixel((i, j))
                XR += r * Sx[i - (x - 1), j - (y - 1)]
                YR += r * Sy[i - (x - 1), j - (y - 1)]
        Mag = math.sqrt(XR * XR + YR * YR)
        if Mag <= D0:
            Mag = 0
        else:
            Mag = 255
        EdgeSobel.putpixel((x, y), (Mag, Mag, Mag))

imgEdgeSobel = np.array(EdgeSobel)

# Tính biên bằng Canny
imgEdgedCanny = cv2.Canny(imgAnhGray, 30, 200)

# Chuyển ảnh gốc về numpy array để hiển thị
imgHinhGoc = np.array(imgHinhGocPIL)

##########################################
#                HIỂN THỊ ẢNH            #
##########################################
cv2.imshow('Hinh Anh Goc', imgHinhGoc)
cv2.imshow('Hinh Anh Gray', imgAnhGray)
cv2.imshow('Nhan Dang Duong Bien Dung Phuong Phap Sobel', imgEdgeSobel)
cv2.imshow('Nhan Dang Duong Bien Dung Phuong Phap Canny', imgEdgedCanny)

cv2.waitKey(0)
cv2.destroyAllWindows()