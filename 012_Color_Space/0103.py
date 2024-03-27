import cv2 as cv
import numpy as np


# --- functions ---

def nothing(value):
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('G', 'image')
    b = cv.getTrackbarPos('B', 'image')
    s = cv.getTrackbarPos(switch, 'image')

    if s == 0:
        img[:] = 0
        # reset trackbars
        # cv.setTrackbarPos('R', 'image', 0)
        # cv.setTrackbarPos('G', 'image', 0)
        # cv.setTrackbarPos('B', 'image', 0)
    else:
        img[:] = [b, g, r]


# --- main ---

img = np.array((512, 512, 3), np.uint8)

cv.namedWindow("image")

cv.createTrackbar('R', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('B', 'image', 0, 255, nothing)

switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image', 1, 1, nothing)

while True:
    cv.imshow('image', img)

    key = cv.waitKey(1) & 0xFF

    if key == 27:
        break

cv.destroyAllWindows()
