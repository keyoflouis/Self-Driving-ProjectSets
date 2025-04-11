import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取标定参数
dist_pickle = pickle.load(open("calibration.pkl", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# 读取图像
img = cv2.imread('IGNORE/calibration_wide/GOPR0069.jpg')
nx = 8  # x方向内角点数量
ny = 6  # y方向内角点数量

#
# def corners_unwarp(img, nx, ny, mtx, dist):
#     # 1) 去畸变
#     undistorted = cv2.undistort(img, mtx, dist, None, mtx)
#
#     # 2) 转换为灰度图
#     gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
#
#     # 3) 检测棋盘角点
#     ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#
#     if ret:
#         # 绘制角点
#         img = cv2.drawChessboardCorners(undistorted.copy(), (nx, ny), corners, ret)
#
#         # 4) 选取四个角点（左上、右上、左下、右下）
#         top_left = corners[0]
#         top_right = corners[nx - 1]
#         bottom_left = corners[(ny - 1) * nx]
#         bottom_right = corners[(ny - 1) * nx + nx - 1]
#         src = np.float32([top_left, top_right, bottom_left, bottom_right])
#
#         # 5) 定义目标点（假设变换到500x500区域）
#         w, h = 500, 500
#         dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
#
#         # 计算透视变换矩阵
#         M = cv2.getPerspectiveTransform(src, dst)
#
#         # 执行透视变换
#         warped = cv2.warpPerspective(undistorted, M, (w, h))
#     else:
#         M = None
#         warped = np.copy(undistorted)
#
#     return warped, M


def corners_unwarp(img, nx, ny, mtx, dist):
    # 使用OpenCV的undistort()函数消除镜头畸变
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # 将校正后的图像转换为灰度图
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # 在灰度图中搜索棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)

        # perspective 变换后边界与图片的距离与尺寸
        offset = 200
        img_size = (gray.shape[1], gray.shape[0])

        # 源点：提取检测到的四个点（A，B，C，D）
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])

        # dst：设置变换后（A，B，C，D）在原图大小的图片中的位置
        dst = np.float32([[offset, offset],
                         [img_size[0]-offset, offset],
                         [img_size[0]-offset, img_size[1]-offset],
                         [offset, img_size[1]-offset]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, M, img_size)

    return warped, M

# 执行变换并显示结果
top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('原始图像', fontsize=50)
ax2.imshow(cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB))
ax2.set_title('校正和透视变换后的图像', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()