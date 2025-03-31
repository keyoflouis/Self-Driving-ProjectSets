import cv2
import pickle
import numpy as np
import glob

# 生成表格坐标，48行，3列，代表48个点，每个点的坐标是xyz
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

print(objp)

objpoints = []
imgpoints = []

images = glob.glob('IGNORE/calibration_wide/GO*.jpg')

gray_shape = []
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = (img.shape[1], img.shape[0])
    # opencv要求输入图片的尺寸是宽，高。因此调节shape

    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (8, 6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tves = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
# ret: 重投影误差（衡量标定精度）。
# mtx: 相机内参矩阵（包含焦距、主点等）。
# dist: 畸变系数，格式为 [k1, k2, p1, p2, k3]（径向畸变k1/k2/k3，切向畸变p1/p2）。
# rvecs, tvecs: 每张图像的外参（旋转和平移向量）。

with open('calibration.pkl', 'wb') as f:
    pickle.dump({"mtx": mtx, "dist": dist}, f)

import pickle
import cv2

with open("calibration.pkl", 'rb') as f:
    calib_data = pickle.load(f)

mtx = calib_data["mtx"]
dist = calib_data["dist"]

print("Distortion coefficients (k1, k2, p1, p2, k3):")
print(dist)

img = cv2.imread("IGNORE/calibration_wide/GOPR0036.jpg")
h, w = img.shape[:2]

dst = cv2.undistort(img,mtx,dist,None,mtx)

cv2.imshow("undistorted",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
