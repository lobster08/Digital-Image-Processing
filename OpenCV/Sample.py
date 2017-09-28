import numpy as np
import cv2

def resample(image, fx, fy):

    tempImage = [image.shape[0]*image.shape[1]]

    width = image.shape[0]
    height = image.shape[1]

    newWidth = width*fx
    newHeight = height*fy

    blank_image = np.zeros((newHeight, newWidth, 3), np.uint8)

    x_ratio = float((width-1)/newWidth)
    y_ratio = float((height-1)/newHeight)

    offset = 0

    for i in range(newHeight):
        for j in range(newWidth):
            x = int(x_ratio * j)
            y = int(y_ratio * i)

            x_diff = (x_ratio * j) - x
            y_diff = (y_ratio * i) - y

            index = (y * width + x)


            a = image[index] & 0xff
            b = image[index+1] & 0xff
            c = image[index+width] & 0xff
            d = image[index+width+1] & 0xff

            gray = int(a*(1-x_diff)*(1-y_diff)+b*(x_diff)*(1-y_diff)+c*(y_diff)*(1-x_diff)+d*(x_diff*y_diff))

            offset += 1
            blank_image[offset] = gray

    return blank_image


def load_display(input_image):
    #pic = cv2.imread(input_image)
    cv2.namedWindow("Lenna", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Lenna", input_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Lenna")

kenny = cv2.imread("kennysmall.jpg",0)
load_display(resample(kenny, 1, 1))