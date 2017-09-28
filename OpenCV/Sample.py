# Create Image
###############
import numpy as np
import cv2

#Display image
def load_display(input_image):
    #pic = cv2.imread(input_image)
    cv2.namedWindow("Lenna", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Lenna", input_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Lenna")

#load_display("Lenna.png")

def create_black_image():
    black_image = np.zeros((300,300), np.uint8)
    return black_image

#load_display(create_black_image())


def create_white_image():
    white_image = np.ones((300,300), np.uint8)*255
    return white_image

#load_display(create_white_image())

def load_display1():

    #reading image
    lenna = cv2.imread("Lenna.png")

    #Create a window to display image
    cv2.namedWindow("Lenna", cv2.WINDOW_AUTOSIZE)

    #display image
    cv2.imshow("Lenna", lenna)

    #This function should be followed by waitKey function which displays the image for specified milliseconds.
    # Otherwise, it won’t display the image.
    cv2.waitKey(0)

    #destroy windows
    cv2.destroyWindow("Lenna")


#Display image size (number)
def image_shape():
    # reading image
    lenna = cv2.imread("Lenna.png")

    lenna = cv2.imread("Lenna.png", 0)

    #gets image shape
    print(lenna.shape)
#image_shape()


#Convert image to grey scale
def convert_to_greyscale():
    # reading image
    lenna = cv2.imread("Lenna.png")

    lenna_grey = cv2.cvtColor(lenna, cv2.COLOR_RGB2GRAY)

    cv2.imwrite("lenna_grey.png", lenna_grey)
    # Create a window to display image
    cv2.namedWindow("Lobster", cv2.WINDOW_AUTOSIZE)

    # display image
    cv2.imshow("Lobster", lenna_grey)

    # This function should be followed by waitKey function which displays the image for specified milliseconds.
    # Otherwise, it won’t display the image.
    cv2.waitKey(0)

#convert_to_greyscale()

#Create a white box on a black background
def create_white_box_black_background():
    box = np.zeros((200, 200), np.uint8)
    shape = box.shape

    #for i in range(shape[0]):
       # for j in range(shape[1]):
         #   if i in range(10, 100) and j in range(10,100):
           #     box[i,j] = 255
    print(shape[0], shape[1])

    return box

load_display(create_white_box_black_background())


#Display image in specific size
def roi_image():
    kennysmall = cv2.imread("kennysmall.jpg")
    kennyface = kennysmall[10:400, 10:500]          #Change size here [Y:y, X:x]
    load_display(kennyface)

#roi_image()

#Change the image's brightness
#Can't greater than 255 or less than 0
def change_image_brightness(image, parameter):
    output_image = image.copy()

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            adjusted = (parameter + output_image[row, col])
            if adjusted > 255.0:
                adjusted = 255.0
            if adjusted < 0:
                adjusted = 0
            output_image[row, col] = np.uint8(adjusted)

    return output_image

#lenna = cv2.imread("Lenna.png", 0)
#load_display(change_image_brightness(lenna, 5))


#Invert image's color
def invert_image(image):
    output_image = image.copy()

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
                output_image[row, col] = 265 - output_image[row, col]

    return output_image

#lenna = cv2.imread("Lenna.png", 0)
#load_display(invert_image(lenna))


#Draw a box onto the image
def draw_rect():
    kenny = cv2.imread("kennysmall.jpg", 0)

    cv2.rectangle(kenny, (90,125),(250,325), (50), thickness=2)
    load_display(kenny)

#draw_rect()

#Draw an ellipse onto the image
def draw_ellipse():
    kenny = cv2.imread("kennysmall.jpg",1)

    cv2.circle(kenny, (170, 230), 100, 10, thickness=3)
    load_display(kenny)

#draw_ellipse()

#Write a text onto the image
def draw_text():
    kenny = cv2.imread("kennysmall.jpg", 0)
    cv2.rectangle(kenny, (90, 125), (250, 325), (255), thickness=5)
    cv2.putText(kenny, "Kenny", (70, 125), cv2.FONT_HERSHEY_COMPLEX, 2, 255)
    load_display(kenny)

#draw_text()

#Display histogram of the image
import matplotlib.pyplot as plt
def histogram(image):

    (row, col) = image.shape
    hist = [0]*256

    for i in range(row):
        for j in range(col):
            hist[image[i,j]] += 1

    return hist

kenny = cv2.imread("kennysmall.jpg", 0)
image_hist1 = histogram(kenny)
image_hist = histogram(invert_image(kenny))

#plt.plot(image_hist1)
#plt.plot(image_hist)
#plt.show()


def resample_times_two(image):
    resampled_image = cv2.resize(image, (0, 0), fx=1.5,fy=1.5, interpolation=cv2.INTER_NEAREST)
    print(resampled_image.shape)
    return resampled_image

#kenny = cv2.imread("kennysmall.jpg",0)
#print(kenny.shape)
#load_display(resample_times_two(kenny))