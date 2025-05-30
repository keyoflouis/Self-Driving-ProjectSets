import os
import glob
import pickle
import numpy as np

import cv2
from skimage.feature import hog
from scipy.ndimage import label

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy import  *
from IPython.display import HTML

def extract_hog(image, pix_per_cell = 12, cell_per_block = 2, orient = 12, vis=False):
    # 存储特征和可视化图像的容器
    hog_features = []
    hog_images = []
    for channel in range(image.shape[2]):
        # 从每个通道中提取 HOG 特征
        if vis:
            hog_feature, hog_image = hog(image[:, :, channel], orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualize=vis, feature_vector=True, transform_sqrt=False)
            # 将新的特征向量追加到特征列表中
            hog_features.append(hog_feature)
            hog_images.append(hog_image)

        else:
            hog_feature = hog(image[:, :, channel], orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualize=vis, feature_vector=True, transform_sqrt=False)
            # 将新的特征向量追加到特征列表中
            hog_features.append(hog_feature)

    # 返回 HOG 图像和特征向量
    return np.concatenate(hog_features), hog_images

def color_hist(img, nbins=64, bins_range=(0, 256)):
    # 计算图像的直方图
    hist_features = np.histogram(img, bins=nbins, range=bins_range)
    # 返回特征向量
    return hist_features[0]

def extract_features(image):
    # 读取图像
    if type(image) == str:
        image = mpimg.imread(image)

    # 将图像转换为 HSV 和 YUV 空间
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # 提取特征
    hog_features = extract_hog(yuv)[0]
    hist_features = color_hist(hsv[:, :, 0])

    # 追加当前图像的特征
    features = np.concatenate((hog_features, hist_features))
    # 返回特征向量列表
    return features

if __name__ =="__main__":

    compare = True

    if compare:
        vehicle_test1 = mpimg.imread("../IGNORE/non-vehicles/Extras/extra1.png")
        vehicle_test2 = mpimg.imread("../IGNORE/non-vehicles/Extras/extra2.png")
        non_vehicle_test1 = mpimg.imread("../IGNORE/vehicles/GTI_MiddleClose/image0000.png")
        non_vehicle_test2 = mpimg.imread("../IGNORE/vehicles/GTI_MiddleClose/image0002.png")
        images = [vehicle_test1, vehicle_test2, non_vehicle_test1, non_vehicle_test2]

        fig, axes = plt.subplots(len(images), 6, figsize=(18, 3 * len(images)))
        for idx, image in enumerate(images):
            # 转换为 HSV 和 YUV 空间
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            # 提取 HSV 特征
            h_features = color_hist(hsv[:, :, 0])
            # 提取 YUV HOG 特征
            yuv_hog_features, yuv_hog_images = extract_hog(yuv, vis=True)
            # 显示
            axes[idx, 0].set_title("origin")
            axes[idx, 0].imshow(image)
            axes[idx, 1].set_title("HSV:H histgram")
            axes[idx, 1].plot(h_features)
            axes[idx, 2].set_title("YUV image")
            axes[idx, 2].imshow(yuv)
            axes[idx, 3].set_title("YUV: Y HOG")
            axes[idx, 3].imshow(yuv_hog_images[0], cmap='gray')
            axes[idx, 4].set_title("YUV: U HOG")
            axes[idx, 4].imshow(yuv_hog_images[1], cmap='gray')
            axes[idx, 5].set_title("YUV: V HOG")
            axes[idx, 5].imshow(yuv_hog_images[2], cmap='gray')
        plt.tight_layout()
        plt.show()