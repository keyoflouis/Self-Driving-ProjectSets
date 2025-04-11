import glob
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

def gene_pkl():
    ''' 生成pkl文件存储相机参数 '''
    obj = np.zeros(shape=(6 * 9, 3), dtype=np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    images = glob.glob("IGNORE/camera_cal/calibration*.jpg")

    image_points = []
    obj_points = []

    gray_shape = []
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_shape = (img.shape[1], img.shape[0])
        ret, corners = cv2.findChessboardCorners(gray_img, (9, 6), None)
        print(f'{idx},{ret}')

        if ret == True:
            obj_points.append(obj)
            image_points.append(corners)

            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tves = cv2.calibrateCamera(obj_points, image_points, gray_shape, None, None)

    with open('IGNORE/output_images/calibration.pkl', 'wb') as f:
        pickle.dump({"mtx": mtx, "dist": dist}, f)



def calibrate(img):
    ''' 读取图片，返回相机矫正后的RGB图片 '''

    with open('IGNORE/output_images/calibration.pkl', 'rb') as f:
        cal = pickle.load(f)

    mtx = cal["mtx"]
    dist = cal["dist"]

    dst = cv2.undistort(img,mtx,dist,None,mtx)

    if __name__ == "__main__":
        # 返回f,两个对象（存放图片）
        f,(row_img, cal_img)=plt.subplots(1, 2)
        f.tight_layout()

        row_img.set_title("row img")
        row_img.imshow(img)
        cal_img.set_title("cal img")
        cal_img.imshow(dst)
        plt.show()

    return dst

if __name__ =="__main__":

    # path = "IGNORE/test_images/test6.jpg"
    # calibrate(path)
    gene_pkl()