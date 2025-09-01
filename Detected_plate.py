import cv2 as cv
import numpy as np
import Pre_process as process
import  main
import PossibleLetter
import DetectLetter
import  Possible_Plate as plate
import math
import random
PaddingHeight = 1.5
PaddingWidth = 1.3
def DetectPlateInScene(imgOriginal):
    listOfPossiblePlate = []
    imgMaxContrast, imgThresh =  process.ImageThresh(imgOriginal)
    imgThreshcopy = imgThresh.copy()
    listOfPossibleCharInScene = FindPossibleCharInScene(imgThreshcopy)
    contours =[]
    height, width = imgThresh.shape
    listOfListsOfMatchingCharsInScene = DetectLetter.find_list_of_lists_matching_letters(listOfPossibleCharInScene)
    # print(len(listOfListsOfMatchingCharsInScene))
    #imgContours = np.zeros((height, width, 3), np.uint8)
    # contours = []
    #
    # for possibleChar in listOfPossibleCharInScene:
    #     contours.append(possibleChar.contour)
    # # end for
    #
    # cv.drawContours(imgContours, contours, -1, main.SCALAR_WHITE)
    # cv.imshow("img", imgContours)
    # cv.waitKey()
    imgContours = np.zeros((height, width, 3), np.uint8)

    #for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
    intRandomBlue = random.randint(0, 255)
    intRandomGreen = random.randint(0, 255)
    intRandomRed = random.randint(0, 255)

    contours = []

    # for matchingChar in listOfListsOfMatchingCharsInScene[2] :
    #     contours.append(matchingChar.contour)
    #     # end for
    #
    # cv.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
    # # end for
    #
    # cv.imshow("3", imgContours)

    for listofMatchingChar in listOfListsOfMatchingCharsInScene:
        PossiblePlate =  ExtraPlate(listofMatchingChar,imgOriginal)
        if PossiblePlate.imgPlate is not None:
            listOfPossiblePlate.append(PossiblePlate)
    # print("\n")
    # cv.imshow("4a", imgContours)
    # print("possibePlate:",len(listOfPossiblePlate))
    # for i in range(0, len(listOfPossiblePlate)):
    #     p2fRectPoints = cv.boxPoints(listOfPossiblePlate[i].rrLocationOfPlateInScene)
    #
    #     cv.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), main.SCALAR_RED, 2)
    #     cv.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), main.SCALAR_RED, 2)
    #     cv.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), main.SCALAR_RED, 2)
    #     cv.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), main.SCALAR_RED, 2)
    #
    #     cv.imshow("imgContour", imgContours)
    #
    #     print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")
    #     # if(listOfPossiblePlates[i].strChars == ""):
    #     #     print("does not exist chars")
    #     # if (listOfPossiblePlates[i].imgThresh is None):
    #     #     print("does not exist thresh")
    #     # if (listOfPossiblePlates[i].imgGrayscale is None):
    #     #     print("does not exist GrayScale")
    #     cv.imshow("imgThresh",imgThresh)
    #     cv.imshow("imgPlateAfterCropped", listOfPossiblePlate[i].imgPlate)
    #     cv.waitKey(0)
    # end for

    # print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
    # cv.waitKey()
    return listOfPossiblePlate
def FindPossibleCharInScene(ImgThresh):
    PossibleCharList = []
    possibleCharcounter = 0 # count the numbers of possible chars
    # find all the contours
    height,width = ImgThresh.shape
    imgThreshcopy = ImgThresh.copy()
    # cv.imshow("imgthreshcopy:",imgThreshcopy)
    contours, hierachies = cv.findContours(imgThreshcopy,cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        possibleChar = PossibleLetter.PossibleLetter(contours[i])
        if(DetectLetter.check_possible_letter(possibleChar)):
            possibleCharcounter=possibleCharcounter+1
            PossibleCharList.append(possibleChar)

    return  PossibleCharList
def ExtraPlate(listOfMatchingChar, imgOriginal):
    PossiblePlate = plate.PossiblePlate()
    list = listOfMatchingChar
    list.sort(key = lambda MatchingChar: MatchingChar.rect_centerx)
    #listOfMatchingChar = sorted(listOfMatchingChar, key = lambda MatchingChar: MatchingChar.rect_centerx)
    #calculate the center point of the plate
    center_point_x = (list[0].rect_centerx+list[len(list)-1].rect_centerx)/2
    center_point_y = (list[0].rect_centery + list[len(list) - 1].rect_centery) / 2
    center_point = center_point_x,center_point_y
    # print("x,y :",center_point_x,center_point_y)
    #calculate width and height
    width_of_plate =  int((list[len(list)-1].rect_width + list[len(list)-1].rect_x - list[0].rect_x)*PaddingWidth)
    total_height = 0
    # print(len(listOfMatchingChar))
    for matchingChar in listOfMatchingChar:
        total_height = total_height + matchingChar.rect_height

        #print("cordinate",matchingChar.rect_height)
    averageHeight = total_height/(len(list))
    height_of_plate = int(averageHeight*PaddingHeight)

    #calculate the angle
    LenghtOfOpposite = list[len(list)-1].rect_centery - list[0].rect_centery
    lenghtOfHypotenuse = DetectLetter.distance_between_letters(list[len(list)-1],list[0])
    AngleInRadian = math.asin(LenghtOfOpposite/lenghtOfHypotenuse)
    AngleInDegree =  AngleInRadian*(180.0/math.pi)
    # print("Opposite: ", lenghtOfHypotenuse)
    # print("Hypotenuse: ", lenghtOfHypotenuse)
    # print("Radian: ", AngleInRadian)
    # print("angle: ",AngleInDegree)
    PossiblePlate.rrLocationOfPlateInScene = ((tuple(center_point)),(width_of_plate,height_of_plate),AngleInDegree)
    rotationMatrix = cv.getRotationMatrix2D(tuple(center_point), AngleInDegree, 1.0)

    height, width, numChannels = imgOriginal.shape  # unpack original image width and height

    imgRotated = cv.warpAffine(imgOriginal, rotationMatrix, (width, height))  # rotate the entire image

    imgCropped = cv.getRectSubPix(imgRotated, (width_of_plate, height_of_plate), tuple(center_point))

    PossiblePlate.imgPlate = imgCropped  # copy the cropped plate image into the applicable member variable of the possible plate
    return PossiblePlate