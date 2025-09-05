import cv2 as cv
import easyocr


def main():

    imgOG = cv.imread("data/new/1.1.PNG",1)
    grayscale = cv.cvtColor(imgOG, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grayscale, (5,5), 0)
    edged = cv.Canny(blurred, 10, 200)

    contours, _ = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:10]

    for c in contours:
        perimeter = cv.arcLength(c, True)
        approximation = cv.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approximation) == 4: # rectangle
            number_plate_shape = approximation
            break
    
    (x, y, w, h) = cv.boundingRect(number_plate_shape)
    img_crop = grayscale[y:y + h, x:x + w]

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(img_crop)
    if result:
        full_text = " ".join([res[-2] for res in result])
        print("=======: " + full_text)
    else:
        print("Empty")

    cv.imshow("imgOG", imgOG)
    cv.imshow("grayscale", grayscale)
    cv.imshow("blurred", blurred)
    cv.imshow("edged", edged)
    cv.imshow("number_plate", img_crop)

    
if __name__ == '__main__':
    main()
    cv.waitKey()