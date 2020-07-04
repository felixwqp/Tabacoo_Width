
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology,draw
from skimage.morphology import medial_axis
from collections import defaultdict
from remove_burr import Clain

"""


"""


def get_width(skel_point, contour, skel):
    # get_slope
    return
    # extend normal to get two point

    # two point distance.




def VThin(image, array):
    h,w= image.shape[:2]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j-1] + image[i,j] + image[i, j+1] if 0<j<w-1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if-1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k, j-1+l] == 255:
                                a[k*3 + l] = 1
                    sum = a[0]*1 + a[1]*2 + a[2]*4 + a[3]*8 + a[5]*16 + a[6]*32 + a[7]*64 + a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    h, w = image.shape[:2]
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i-1, j] + image[i, j] + image[i+1, j] if 0<i<h-1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k, j-1+l] == 255:
                                a[k*3 + l] = 1
                    sum = a[0]*1 + a[1]*2 + a[2]*4 + a[3]*8 + a[5]*16 + a[6]*32 + a[7]*64 + a[8]*128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def Xihua(binary, array, num=10):
    iXihua = binary.copy()
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)
    return iXihua


array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


image = cv.imread("image0.bmp")

# convert to RGB
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# create a binary thresholded image
_, binary = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)
# show it
# plt.imshow(binary, cmap="gray")
# plt.show()


# find the contours from the thresholded image
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# draw all contours
# print(len(contours))
max_area = -1
max_area_idx = 0
area_record = [0 ]* len(contours)
for i in range(len(contours)):
    temp_area =cv.contourArea(contours[i])
    area_record[i] = temp_area
    if max_area < temp_area:
        max_area = temp_area
        max_area_idx = i

new_contours = []

for i in range(len(contours)):
    if area_record[i] >= max_area * 0.1 and area_record[i] <= max_area * 0.2:
        cnt = contours[i]
        new_contours.append(cnt)
        width_cnt = cv.contourArea(cnt) / cv.arcLength(cnt, True) * 2
        print('Avg Width: Area/arcLength * 2 : ', width_cnt)

        # rect = cv.minAreaRect(cnt)
        # box = cv.boxPoints(rect)
        # box = np.int0(box)
        # print(box)
        # cv.drawContours(img, [box], 0, (0, 0, 255), 2)
        # print('M: ',i)
        # print(cv.moments(contours[i]))
# empty_image
image[...] = 0

image = cv.drawContours(image, new_contours, -1,   (0, 255, 0), cv.FILLED)


# convert to RGB
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# create a binary thresholded image
print(gray.shape)
print(image.shape)
# _, binary = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)


blur=((5,5),1)
erode_=(10,10)
dilate_=(20, 20)
gray = cv.erode(cv.dilate(cv.GaussianBlur(gray, blur[0], blur[1]), np.ones(dilate_)), np.ones(erode_) , 1)



# plt.imshow(gray1)
# plt.show()
# plt.imshow(gray4)
# plt.show()


# skeleton =morphology.skeletonize(gray, method='lee')

# print(len(contours))
# show the image with the drawn contours


# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(gray, return_distance=True)
dist_on_skel = distance * skel


# erode and dilate for skel
erode_skel_20 =(15, 15)
erode_skel_10 =(10, 10)
dilate_skel = (10, 10)

skel_float = skel.astype(float)

skel_dilate = cv.dilate(skel_float, np.ones(dilate_skel))
skel_erode_20  = cv.erode(skel_dilate, np.ones(erode_skel_20))
skel_erode_10  = cv.erode(skel_dilate, np.ones(erode_skel_10))

plt.imshow(skel_dilate)
plt.imshow(skel_erode_20)
plt.imshow(skel_erode_10)
plt.imshow(skel)
plt.show()




x = Clain()
x.selct(skel)
x.start()
last = x.dpoint
while True:
    x.selct(x.imag)
    now=x.dpoint
    x.start()
    if now==last:
        break
    else:
        last=now

x1 = x.im
x2 = x.imag


print(dist_on_skel.shape)

points = defaultdict(list)#[[x, y, val],]
for i in range(len(dist_on_skel)):
    vals = dist_on_skel[i]
    for j in range(len(vals)):
        val = dist_on_skel[i][j]
        if val > 1:
            # print(i, j, distance[i][j])
            for idx_cnt in range(len(new_contours)):
                if cv.pointPolygonTest(new_contours[idx_cnt], (i, j), True):
                    # print(idx_cnt)
                    points[idx_cnt].append([i,j,val])
                    break

# plt.imshow(skel)
print(skel.shape)
print(distance.shape)



# extent = np.min(x), np.max(x), np.min(y), np.max(y)
fig = plt.figure(frameon=False)


im1 = plt.imshow(gray, cmap=plt.cm.gray, interpolation='nearest')


im2 = plt.imshow(skel, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear')

plt.show()
#
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,  figsize=(12, 4))
#
#
# ax1.imshow(skel, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.set_title('skeleton', fontsize=20)
#
# ax2.imshow(distance, cmap=plt.cm.gray)
# ax2.axis('off')
# ax2.set_title('distance', fontsize=20)
#
# ax3.imshow(dist_on_skel, cmap=plt.cm.gray)
# ax3.axis('off')
# ax3.set_title('dist on skel', fontsize=20)
#
# fig.tight_layout()
# plt.show()
