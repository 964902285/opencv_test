import cv2


# --- functions ---

def function(value):
    print(value)

    new_img = img.copy()
    new_img[:, :, 0] = value  # B G R

    cv2.imshow('image dst', new_img)


# --- main ---

img = cv2.imread('../imgs/src2.jpg')
cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 0, 255, function)
cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
