"""链码去除骨刺"""
import numpy as np
import cv2
from skimage import morphology, draw


class Clain():
    def __init__(self):
        self.jpoint = []  # 节点
        self.dpoint = []  # 端点

    def selct(self, img):
        self.imag = img.copy()
        self.im = img.copy()
        self.m, self.n = img.shape
        for k in range(self.m):
            for r in range(self.n):
                if img[k, r] == 255:
                    self.point(k, r, img)

    def point(self, y, x, img):
        try:
            s1 = int(img[y, x])
            s2 = int(img[y - 1, x - 1])
            s3 = int(img[y - 1, x])
            s4 = int(img[y - 1, x + 1])
            s5 = int(img[y, x - 1])
            s6 = int(img[y, x + 1])
            s7 = int(img[y + 1, x - 1])
            s8 = int(img[y + 1, x])
            s9 = int(img[y + 1, x + 1])
            dian = (int((s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9) / 255))
            if dian >= 4:
                self.jpoint.append([y, x])  # 节点
            if dian == 2:
                self.dpoint.append([y, x])
        except:
            print('边界error')

    def start(self):
        for p in self.dpoint:
            self.l = []
            self.a, self.b = p[0], p[1]
            self.im[p[0], p[1]] = 0
            self.code(p[0], p[1])

    def code(self, y, x):
        try:
            l = [self.im[y, x + 1], self.im[y - 1, x + 1], self.im[y - 1, x], self.im[y - 1, x - 1], self.im[y, x - 1],
                 self.im[y + 1, x - 1], self.im[y + 1, x], self.im[y + 1, x + 1]]
            if not sum(l) == 0 and sum(l) < 510:
                s = l.index(255)
                self.l.append(s)
                dic = {0: (y, x + 1), 1: (y - 1, x + 1), 2: (y - 1, x), 3: (y - 1, x - 1), 4: (y, x - 1),
                       5: (y + 1, x - 1), 6: (y + 1, x), 7: (y + 1, x + 1)}
                self.im[y, x] = 0
                a, b, = dic[s]
                if [a, b] not in self.jpoint:
                    self.code(a, b)
                else:
                    if len(self.l) < 12:  # 对端点到节点距离小于12个像素的进行去除（认为是骨刺）
                        y = self.a
                        x = self.b
                        self.imag[y, x] = 0
                        for h in self.l:
                            dic = {0: (y, x + 1), 1: (y - 1, x + 1), 2: (y - 1, x), 3: (y - 1, x - 1), 4: (y, x - 1),
                                   5: (y + 1, x - 1), 6: (y + 1, x), 7: (y + 1, x + 1)}  # 利用字典来编码
                            y, x = dic[h]
                            self.imag[y, x] = 0
        except:
            pass
            # print(666)
        # return  img


if __name__ == '__main__':
    img = cv2.imread('test.jpg', 0)  # 读取图片
    ret, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)  # 二值化
    img = morphology.skeletonize_3d(img)  # 提取骨架
    x = Clain()
    x.selct(img)
    x.start()
    last = x.dpoint
    """如果前后两次的端点一样，说明骨刺去除完毕"""
    while True:
        x.selct(x.imag)
        now = x.dpoint
        x.start()
        if now == last:
            break
        else:
            last = now
    cv2.imshow('img', img)
