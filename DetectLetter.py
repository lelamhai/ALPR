import cv2 as cv
import numpy as np
import Pre_process
import  main
import math
import os

import cv2 as cv
import numpy as np

import PossibleLetter

#################################
#   GLOBAL VAR
# sử dụng KNN
KNEAREST = cv.ml.KNearest_create()

# Thông số để kiểm tra các đặc điểm có phải là một ký tự KHẢ NĂNG trong ảnh hay không
MIN_WIDTH = 2
MIN_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_AREA = 80

#  Thông số để kiểm tra các đặc điểm giữa 2 kí tự xem chung có TƯƠNG ĐỒNG
#   VD : kiểu số trên biển số xe sẽ có tương tự kiểu dáng về khoảng cách giữa các số , thẳng hàng và ngăn nắp trên biển
#  ==> được tính theo tỉ lệ trên kí tự đem so sánh
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0
# Nếu ít nhất ko có 4 số TƯƠNG ĐỒNG thì không có khả năng là biển số
MIN_NUMBER_OF_MATCHING_CHARS = 4

# Dùng để resize ảnh của chữ để train KNN
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


###########################################
# load KNN classification vs training data (flattened img)
def load_and_train_knn():
    # load the labels = classifications
    try:
        classifications_np = np.loadtxt("classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")  # show error message
        os.system("pause")
        return False

    try:
        flattened_img = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")  # show error message
        os.system("pause")
        return False

    #  reshape the test data = classifications
    classifications = classifications_np.reshape(classifications_np.size, 1)

    # train  data with KNN
    KNEAREST.train(flattened_img, 0, classifications)

    # true of we train successfully
    return True


############################################
#   FUNCTION
def check_possible_letter(possible_letter):
    if possible_letter.rect_width > MIN_WIDTH \
            and possible_letter.rect_height > MIN_HEIGHT \
            and possible_letter.rect_area > MIN_AREA \
            and MIN_ASPECT_RATIO < possible_letter.rect_aspectratio < MAX_ASPECT_RATIO:
        return True
    else:
        return False


def find_possible_letter(img_thresh):
    # contour that pass the check_possibleLetter in here
    list_possibleletter = []

    img_thresh_cop = img_thresh.copy()

    contours, hierarchy = cv.findContours(img_thresh_cop, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possible_letter = PossibleLetter.PossibleLetter(contour)

        if check_possible_letter(possible_letter):
            list_possibleletter.append(possible_letter)

    return list_possibleletter


def distance_between_letters(letter1, letter2):
    # Sử dụng công thức pythago để tính khoảng cách
    # int_x , int_y là 2 cạnh góc vuông đc tính như sau:
    int_x = abs(letter2.rect_centerx - letter1.rect_centerx)
    int_y = abs(letter2.rect_centery - letter1.rect_centery)

    distance = math.sqrt((int_x ** 2) + (int_y ** 2))

    return distance


def angle_between_letters(letter1, letter2):
    # Sử dụng công thức lượng giác để tính góc ==> ta sd tan
    # adjacent = cạnh kề , opposite = cạnh đối được tính như sau
    adjacent = float(abs(letter2.rect_centerx - letter1.rect_centerx))
    opposite = float(abs(letter2.rect_centery - letter1.rect_centery))

    # kiểm tra xem đối/kề có nghĩa hay không
    if adjacent != 0.0:
        angle_rad = math.atan(opposite / adjacent)
    else:
        # nếu vô nghĩa t cho angle_rad = tan(1)
        angle_rad = 1.5574

    # đổi từ rad sang deg(độ)
    angle_deg = angle_rad * (180 / math.pi)

    return angle_deg


def find_list_matching_letters(possible_letter, list_of_letters):
    list_matching_letter = []
    # print(possible_letter)
    # print("_______________")
    for possible_matching_letter in list_of_letters:
        if possible_letter == possible_matching_letter:
            continue

        # Tính toán các tỉ lệ (float) đặc điểm của kí tự có thể TƯƠNG ĐỒNG với kí tự đc đem so sánh

        # Tỉ lệ độ lệch trong diện tích
        area_diff = float(abs(possible_matching_letter.rect_area - possible_letter.rect_area)) / float(
            possible_letter.rect_area)
        # Khoảng cách giữa 2 kí tự trên ảnh
        distance_diff = distance_between_letters(possible_letter, possible_matching_letter)
        # Góc lệch giữa 2 kí tự trên ảnh
        angle_diff = angle_between_letters(possible_letter, possible_matching_letter)
        # Tỉ lệ độ lệch về chiều dài
        width_diff = float(abs(possible_matching_letter.rect_width - possible_letter.rect_width)) / float(
            possible_letter.rect_width)
        # Tỉ lệ độ lệch về chiều rộng
        height_diff = float(abs(possible_matching_letter.rect_height - possible_letter.rect_height)) / float(
            possible_letter.rect_height)

        # Kiểm tra có thỏa các thông số đã khai báo
        if (distance_diff < (possible_letter.rect_diagonalsize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                area_diff < MAX_CHANGE_IN_AREA and
                angle_diff < MAX_ANGLE_BETWEEN_CHARS and
                width_diff < MAX_CHANGE_IN_WIDTH and
                height_diff < MAX_CHANGE_IN_HEIGHT):
            list_matching_letter.append(possible_matching_letter)

    return list_matching_letter


def find_list_of_lists_matching_letters(list_possible_letter):
    list_of_lists_matching_letters = []

    for possible_letter in list_possible_letter:
        # print(possible_letter)
        list_matching_letters = find_list_matching_letters(possible_letter, list_possible_letter)

        list_matching_letters.append(possible_letter)

        if len(list_matching_letters) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        list_of_lists_matching_letters.append(list_matching_letters)

        # loại trừ các list chứa các số có khả năng là của biển số
        list_possible_letter_without_matched_letter = list(set(list_possible_letter) - set(list_matching_letters))

        # đệ qui để tiếp tục tìm ra các list khác chứa khả năng là nhóm số của của biển số
        recursive_list_of_lists_matching_letters = find_list_of_lists_matching_letters(
            list_possible_letter_without_matched_letter)

        for recursive_list_of_matching_letters in recursive_list_of_lists_matching_letters:
            list_of_lists_matching_letters.append(recursive_list_of_matching_letters)

        break
    # print(len(list_of_lists_matching_letters))
    # print("_______________")
    return list_of_lists_matching_letters


def remove_inner_overlapping_letters(list_matching_letters):
    # tạo mảng chứa các ký tự không bị trùng lắp lên nhau
    list_matching_letters_without_overlapping = list(list_matching_letters)

    # Duyệt giữa các ký tự trong mảng
    for current_letter in list_matching_letters:
        for compare_letter in list_matching_letters:
            if current_letter != compare_letter:
                # nếu true t hì 2 ký này có khả năng bị trùng lắp lên nhau
                # Kiểm tra xem tâm của 2 ký tự này trên dường chép có trùng nhau hay không
                if distance_between_letters(current_letter,
                                            compare_letter) < (
                        current_letter.rect_diagonalsize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # Nếu trùng thì xét tiếp chữ nào nhỏ hơn thì xét loại bỏ ký tự đó
                    if current_letter.rect_area < compare_letter.rect_area:
                        # Nếu ký tự này chưa được xóa trong mảng list_matching_letters
                        if current_letter in list_matching_letters_without_overlapping:
                            list_matching_letters_without_overlapping.remove(current_letter)
                    else:
                        if compare_letter in list_matching_letters_without_overlapping:
                            list_matching_letters_without_overlapping.remove(compare_letter)

    return list_matching_letters_without_overlapping


def regconize_letters_in_plate(img_thresh, list_matching_letters):
    res_str = ""  # the result number in plates will print in console

    # make a color version of thresh img and we can draw contours in color on it
    thresh_height, thresh_width = img_thresh.shape
    # a black only img
    img_thresh_color = np.zeros((thresh_height, thresh_width, 3), np.uint8)

    # sort the list of letters in order
    list_matching_letters.sort(key=lambda matching_letter: matching_letter.rect_centerx)

    cv.cvtColor(img_thresh, cv.COLOR_GRAY2BGR, img_thresh_color)

    for current_letters in list_matching_letters:
        point1 = (current_letters.rect_x, current_letters.rect_y)
        point2 = ((current_letters.rect_x + current_letters.rect_width),
                  (current_letters.rect_y + current_letters.rect_height))

        # draw a red box around the letter
        # cv.rectangle(img_original, point1, point2, (0, 0, 255), 2)

        # crop out the letter
        img_roi = img_thresh[current_letters.rect_y:current_letters.rect_y + current_letters.rect_height,
                  current_letters.rect_x:current_letters.rect_x + current_letters.rect_width]

        # resize the letter ==> necessary for letters regconition
        img_roi_resized = cv.resize(img_roi, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))

        # cv.imshow("letter", img_roi_resized)
        # cv.waitKey()
        # flatten the img_roi =  img of the letter
        img_roi_flatten = img_roi_resized.reshape(1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT).astype(
            np.float32)

        retvalue, result_np, neighbour, distance = KNEAREST.findNearest(img_roi_flatten, k=1)

        # current letter that regconize
        result = chr(int(result_np[0][0]))
        res_str = res_str + result

    return res_str
def DetectLetterInPlate(ListofPossiblePlate):

    if len(ListofPossiblePlate) == 0:
        return ListofPossiblePlate
    for PossiblePlate in ListofPossiblePlate:
        PossiblePlate.imgGrayscale, PossiblePlate.imgThresh = Pre_process.ImageThresh(PossiblePlate.imgPlate)

        #increase size of the image
        PossiblePlate.imgThresh = cv.resize(PossiblePlate.imgThresh,(0,0),fx=1.9,fy=1.9)
        # cv.imshow("thresh1", PossiblePlate.imgThresh)
        thresholdValue, PossiblePlate.imgThresh = cv.threshold(PossiblePlate.imgThresh, 0.0, 255.0,
                                                                cv.THRESH_BINARY | cv.THRESH_OTSU)
        # print("value:",thresholdValue)
        # cv.imshow("thresh2", PossiblePlate.imgThresh)
        # cv.waitKey()
        #find the possible letter in the plate
        listOfPossibleCharInPlate = find_possible_letter(PossiblePlate.imgThresh)
        # given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = find_list_of_lists_matching_letters(listOfPossibleCharInPlate)

        if(len(listOfListsOfMatchingCharsInPlate )== 0):
            PossiblePlate.strChars = ""
            continue
        for i in range(0,len(listOfListsOfMatchingCharsInPlate)):
            listOfListsOfMatchingCharsInPlate[i].sort(key=lambda matchingChar: matchingChar.rect_centerx)  # sort chars from left to right
            # and remove inner overlapping chars
            listOfListsOfMatchingCharsInPlate[i] = remove_inner_overlapping_letters( listOfListsOfMatchingCharsInPlate[i])
        # end for
        intlenofLongestCharlist = 0
        intIndexofLongestList = 0
        # in this loop we find the longest list in the list of list of matching char because it can be license plate
        for i in range(0,len(listOfListsOfMatchingCharsInPlate)):
            if(len(listOfListsOfMatchingCharsInPlate[i])>intlenofLongestCharlist):
                intIndexofLongestList = i
                intlenofLongestCharlist = len(listOfListsOfMatchingCharsInPlate[i])
        TheLongestList = listOfListsOfMatchingCharsInPlate[intIndexofLongestList]
        PossiblePlate.strChars = regconize_letters_in_plate(PossiblePlate.imgThresh, TheLongestList)
    # print(ListofPossiblePlate[0].strChars)
    return ListofPossiblePlate