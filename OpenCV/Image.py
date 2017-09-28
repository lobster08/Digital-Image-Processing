import numpy as np
import cv2
w, h = 1024, 768

img = np.empty((w,h), np.uint8)
img.shape = h,w

for i in range(h):
    for j in range(w):
        img[i,j] = 255

cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyWindow("Image")