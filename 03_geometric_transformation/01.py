# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Name:         01.py
# Description:  计算仿射矩阵
# Author:       xuxianghang
# Date:         2024/3/27 21:54
# IDE:          PyCharm
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import cv2 as cv
import numpy as np


# ------方程法------
src = np.array(
    [[0, 0],
     [200, 0],
     [0, 200]], dtype=np.float32
)
dst = np.array(
    [[0, 0],
     [100, 0],
     [0, 100]], dtype=np.float32
)
A = cv.getAffineTransform(src, dst)
print(A)

# ------矩阵法------
s = np.array(
    [[0.5, 0, 0],
     [0, 0.5, 0],
     [0, 0, 1]], dtype=np.float32
) # 缩放矩阵
t = np.array(
    [[1, 0, 100],
     [0, 1, 200],
     [0, 0, 1]], dtype=np.float32
) # 平移矩阵
B = np.dot(t, s) # 矩阵相乘
print(B)
# ------对于等比例缩放的仿射变换------
C = cv.getRotationMatrix2D(
    (40, 50), 30, 0.5
)
print(C.dtype)
print(C)
