import cv2


# --- functions ---

def function(index, value):
    percent = (value / 100)
    print(index, value, f"{value}%")

    img[:, :, index] = original_img[:, :, index] * percent

    cv2.imshow('image', img)


# --- main ---

img = cv2.imread('../imgs/src6.jpg')

original_img = img.copy()

cv2.namedWindow('image')

cv2.createTrackbar('R (%)', 'image', 50, 100, lambda value: function(2, value))
cv2.createTrackbar('G (%)', 'image', 50, 100, lambda value: function(1, value))
cv2.createTrackbar('B (%)', 'image', 50, 100, lambda value: function(0, value))

cv2.imshow('image', img)
ch = cv2.waitKey(0)
while True:
    if ch == 13:
        cv2.imwrite(f"img_dst.jpg", img)
        print("saved success!")
        break

# cv2.waitKey(0)
#     if ch == 27:
    cv2.destroyAllWindows()
