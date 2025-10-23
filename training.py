import cv2
import numpy as np
import os

# Đọc ảnh gốc
img = cv2.imread('data/training_chars.png', cv2.IMREAD_GRAYSCALE)

# Nhị phân hóa
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Tìm contour của từng ký tự
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

chars = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    roi = thresh[y:y+h, x:x+w]
    roi = cv2.resize(roi, (28, 28))
    chars.append(roi)

X = np.array(chars).reshape(-1, 28, 28, 1) / 255.0
y = np.array(labels)  # ví dụ [0, 1, 2, ..., 35]