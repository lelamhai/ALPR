import cv2 as cv
import numpy as np
import Detected_plate as plate
import  Pre_process as Pre
import  DetectLetter as letter

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

def main():
    blnKNNTrainingSuccessful = letter.load_and_train_knn()

    imgOG = cv.imread("data/8.png",1)

    imgOG_gray,img_thresh = Pre.ImageThresh(imgOG)
    listOfPossiblePlate = plate.DetectPlateInScene(imgOG)
    # detect the plate in picture

    listOfPossiblePlate = letter.DetectLetterInPlate(listOfPossiblePlate)

    if len(listOfPossiblePlate) == 0:
        print("There's no plate found")
        return
    else:
        listOfPossiblePlate.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        licensePlate= listOfPossiblePlate[0].strChars
        print(licensePlate)
        PicPlate = listOfPossiblePlate[0]
        cv.imshow("imgPlate",PicPlate.imgPlate)
        cv.imshow("imgPlate GrayScale",PicPlate.imgGrayscale)
        cv.imshow("imgPlate Thresh",PicPlate.imgThresh)
        #DrawRectangle(imgOG,PicPlate)
    DrawRectangle(imgOG,PicPlate)
    writeLicensePlateCharsOnImage(imgOG,PicPlate)
    cv.imshow("Image Original", imgOG)
    cv.waitKey()
    # cv.imshow("gray", imgOG_gray)
    # cv.imshow("imageThresh", img_threshcopy)
    # print(len(contours))

def DrawRectangle(imgOG, picPlate):
    ArrPointToDrawRect = cv.boxPoints(picPlate.rrLocationOfPlateInScene,)
    for i in range(0,len(ArrPointToDrawRect)-1):
       cv.line(imgOG,tuple(ArrPointToDrawRect[i].astype(int)),tuple(ArrPointToDrawRect[i+1].astype(int)),SCALAR_RED,2)
    cv.line(imgOG, tuple(ArrPointToDrawRect[3].astype(int)), tuple(ArrPointToDrawRect[0].astype(int)), SCALAR_RED, 2)
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    # this will be the center of the area the text will be written to
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    # choose a plain jane font
    intFontFace = cv.FONT_HERSHEY_SIMPLEX
    # base font scale on height of plate area
    fltFontScale = float(plateHeight) / 30.0
    # base font thickness on font scale
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv.getTextSize(
        licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

    # unpack roatated rect into center point, width and height, and angle
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    # make sure center is an integer
    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    # the horizontal location of the text area is the same as the plate
    ptCenterOfTextAreaX = int(intPlateCenterX)

    # if the license plate is in the upper 3/4 of the image
    if intPlateCenterY < (sceneHeight * 0.75):
        # write the chars in below the plate
        ptCenterOfTextAreaY = int(
            round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    # else if the license plate is in the lower 1/4 of the image
    else:
        # write the chars in above the plate
        ptCenterOfTextAreaY = int(
            round(intPlateCenterY)) - int(round(plateHeight * 1.6))
    # end if

    # unpack text size width and height
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    # based on the text area center, width, and height
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    # write the text on the image
    cv.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX,
                                                      ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
if __name__ == '__main__':
    main()