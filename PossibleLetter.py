import math

import cv2 as cv


###############################
# an obj
class PossibleLetter:
    def __init__(self, input_contour):
        # constructor with an input contour
        self.contour = input_contour
        # character total in a rectangle
        self.bounding_rect = cv.boundingRect(self.contour)

        int_x, int_y, int_width, int_height = self.bounding_rect

        self.rect_x = int_x
        self.rect_y = int_y
        self.rect_width = int_width
        self.rect_height = int_height

        # character total area = rect area
        self.rect_area = self.rect_width * self.rect_height

        # get (x,y) at the center of the rect
        self.rect_centerx = self.rect_x + self.rect_width / 2
        self.rect_centery = self.rect_y + self.rect_height / 2

        # Tính đường chéo của rect sử dụng pitago
        self.rect_diagonalsize = math.sqrt((self.rect_width ** 2) + (self.rect_height ** 2))

        # Tính tỉ lệ của width/height = a float numbers
        self.rect_aspectratio = float(self.rect_width) / float(self.rect_height)
