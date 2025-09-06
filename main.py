import cv2 as cv
import easyocr

DISTANCE = 10.0

def main():

    imgOG = cv.imread("data/16.jpg",1)
    grayscale = cv.cvtColor(imgOG, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grayscale, (5,5), 0)
    edged = cv.Canny(blurred, 10, 200)

    contours, _ = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:10]


    prev_box = None
    for i, c in enumerate(contours):
        peri   = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:                     # ứng viên hình chữ nhật
            x, y, w, h = cv.boundingRect(approx)

            if prev_box is not None:
                px, py, pw, ph = prev_box
                if abs(px - x) < DISTANCE and abs(py - y) < DISTANCE and abs(pw - w) < DISTANCE and abs(ph - h) < DISTANCE:
                    print("Near")
                    continue
            prev_box = x, y, w, h

            # chặn biên cho an toàn
            H, W = imgOG.shape[:2]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)

            roi = imgOG[y0:y1, x0:x1].copy()       # hoặc dùng grayscale nếu muốn ảnh xám
            cv.imshow(f"plate_{i}", roi)        # hiển thị từng crop
            reader = easyocr.Reader(['en'] , gpu=False)
            result = reader.readtext(roi)
            if result:
                full_text = " ".join([res[-2] for res in result])
                print("=======: " + full_text)
            else:
                print("Empty")


    cv.imshow("imgOG", imgOG)
    cv.imshow("grayscale", grayscale)
    cv.imshow("blurred", blurred)
    cv.imshow("edged", edged)

    # plate = []
    # for c in contours:
    #     perimeter = cv.arcLength(c, True)
    #     approximation = cv.approxPolyDP(c, 0.02 * perimeter, True)
    #     if len(approximation) == 4: # rectangle
    #         number_plate_shape = approximation
    #         break


    # (x, y, w, h) = cv.boundingRect(number_plate_shape)
    # img_crop = grayscale[y:y + h, x:x + w]
    # cv.imshow("number_plate", img_crop)
    # reader = easyocr.Reader(['en'] , gpu=False)
    # result = reader.readtext(img_crop)

    # if result:
    #     full_text = " ".join([res[-2] for res in result])
    #     print("=======: " + full_text)
    # else:
    #     print("Empty")

    # cv.imshow("imgOG", imgOG)
    # cv.imshow("grayscale", grayscale)
    # cv.imshow("blurred", blurred)
    # cv.imshow("edged", edged)
    # cv.imshow("number_plate", img_crop)

    # rois = []
    # for i, c in enumerate(contours):
    #     peri   = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    #     if len(approx) == 4:                     # ứng viên hình chữ nhật
    #         x, y, w, h = cv2.boundingRect(approx)

    #         # chặn biên cho an toàn
    #         H, W = img.shape[:2]
    #         x0, y0 = max(0, x), max(0, y)
    #         x1, y1 = min(W, x + w), min(H, y + h)

    #         roi = img[y0:y1, x0:x1].copy()       # hoặc dùng grayscale nếu muốn ảnh xám
    #         rois.append(roi)
    #         cv2.imshow(f"plate_{i}", roi)        # hiển thị từng crop


    
if __name__ == '__main__':
    main()
    cv.waitKey()