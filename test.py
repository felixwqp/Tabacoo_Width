import cv2 as cv
import matplotlib.pyplot as plt


image = cv.imread("image0.bmp")

# convert to RGB
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# create a binary thresholded image
_, binary = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)
# show it

des = cv.bitwise_not(gray)
contour,hier = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)

max_area = -1
max_area_idx = 0
area_record = [0 ]* len(contour)
for i in range(len(contour)):
    temp_area =cv.contourArea(contour[i])
    area_record[i] = temp_area
    if max_area < temp_area:
        max_area = temp_area
        max_area_idx = i

des[...] = 0
for i in range(len(contour)):
    if area_record[i] >= max_area * 0.1 and area_record[i] <= max_area * 0.7:
        cnt = contour[i]
        cv.drawContours(des, [cnt], 0, 255, cv.FILLED)

# for cnt in contour:
#     if
#     cv.drawContours(des,[cnt],0,255,-1)

gray = cv.bitwise_not(des)


plt.imshow(gray)
plt.show()