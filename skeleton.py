# from skimage import morphology,draw
# import numpy as np
# import matplotlib.pyplot as plt
#
# #创建一个二值图像用于测试
# image = np.zeros((400, 400))
#
# #生成目标对象1(白色U型)
# image[10:-10, 10:100] = 1
# image[-100:-10, 10:-10] = 1
# image[10:-10, -100:-10] = 1
#
# #生成目标对象2（X型）
# rs, cs = draw.line(250, 150, 10, 280)
# for i in range(10):
#     image[rs + i, cs] = 1
# rs, cs = draw.line(10, 150, 250, 280)
# for i in range(20):
#     image[rs + i, cs] = 1
#
# #生成目标对象3（O型）
# ir, ic = np.indices(image.shape)
# circle1 = (ic - 135)**2 + (ir - 150)**2 < 30**2
# circle2 = (ic - 135)**2 + (ir - 150)**2 < 20**2
# image[circle1] = 1
# image[circle2] = 0
#
# #实施骨架算法
# skeleton =morphology.skeletonize(image)
#
# #显示结果
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
#
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.set_title('original', fontsize=20)
#
# ax2.imshow(skeleton, cmap=plt.cm.gray)
# ax2.axis('off')
# ax2.set_title('skeleton', fontsize=20)
#
# fig.tight_layout()
# plt.show()
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

# Invert the horse image
image = invert(data.horse())

# perform skeletonization
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()