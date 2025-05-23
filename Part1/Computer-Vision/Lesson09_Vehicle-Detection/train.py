import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pickle

from sklearn.svm import LinearSVC


def loadDataset(path, target_dirs):
    ''' 返回车辆图片和非车辆图片的路径集合 '''

    images = []
    for dir_name in target_dirs:
        start_path = os.path.join(path, dir_name)
        for root, _, files in os.walk(start_path):
            for file in files:
                if file.lower().endswith(('png')):
                    images.append(os.path.join(root, file))

    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    return cars, notcars


def bin_spatial(img, size=(32, 32)):
    '''
    降采样，展平为一维特征数组
    '''
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    统计图像各通道的分布直方图
    :param nbins:直方数量
    :param bins_range: 只统计阈值内
    '''

    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    chaneel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    return np.concatenate((channel1_hist[0], channel2_hist[0], chaneel3_hist[0]))


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    '''
    1. 对传入的图像像素值开根，
    2. 对图片使用sobel算子求导，
    3. 用单元格集和块划分图像。
    4. 以块为单位，计算每个块内所有cell的直方图，得到一维向量
    5. 计算一维向量的L2范数 L2 = √(v₁² + v₂² + ... + vₙ²)
    6. 对向量进行归一化：v_normalized = v / (L2 + ε) (ε是一个很小的常数，防止除以0)

    :param img: 传入图片
    :param orient: 块内直方图数量
    :param pix_per_cell: 单元格的像素数量=（ pix_per_cell * pix_per_cell ）
    :param cell_per_block: 块内单元格数量
    :param vis: 可视化
    :param feature_vec: 是否将输出向量展平
    :return: 特征图
    '''

    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm="L2-Hys",
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image

    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       block_norm="L2-Hys",
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualize=vis,
                       feature_vector=feature_vec)

        return features


def extract_features(imgs, color_space="RGB", spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0, use_spatial=True, use_colorhist=True, use_hog=True):
    features = []

    for file in imgs:
        file_feature = []
        image = mpimg.imread(file)

        image = (image * 255).astype(np.uint8)

        if color_space != "RGB":
            if color_space == "HSV":
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == "LUV":
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == "HLS":
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == "YUV":
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == "YCrCb":
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if use_spatial == True:
            spatial_feature = bin_spatial(feature_image, spatial_size)
            file_feature.append(spatial_feature)

        if use_colorhist == True:
            hist_feature = color_hist(feature_image, hist_bins)
            file_feature.append(hist_feature)

        if use_hog == True:

            if hog_channel == "ALL":
                hog_feature = []
                for channel in range(feature_image.shape[2]):
                    hog_feature.extend(
                        get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block))
                hog_feature = np.ravel(hog_feature)
            else:
                hog_feature = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block)

            file_feature.append(hog_feature)

        features.append(np.concatenate(file_feature))

    return features


if __name__ == "__main__":
    path = "./IGNORE/"
    target_dirs = {'non-vehicles', 'vehicles'}
    cars, notcars = loadDataset(path, target_dirs)

    print(len(cars),len(notcars))

    # TODO: 调整参数
    color_space = 'YCrCb'
    # hog 参数
    hog_feat = True
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL"
    # 空间特征参数
    spatial_feat = True
    spatial_size = (32, 32)
    # 颜色特征参数
    hist_feat = True
    hist_bins = 32
    # 图片范围
    y_start_stop = [400, None]

    # 提取特征
    car_features = extract_features(cars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block,
                                    hog_channel, use_spatial=spatial_feat, use_colorhist=hist_feat, use_hog=hog_feat)
    notcar_features = extract_features(notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                                       cell_per_block, hog_channel, use_spatial=spatial_feat, use_colorhist=hist_feat,
                                       use_hog=hog_feat)

    X = np.vstack((car_features, notcar_features,)).astype(np.float64)
    Y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=rand_state)

    X_scaler =StandardScaler().fit(X_train)
    X_train =X_scaler.transform(X_train)
    X_test =X_scaler.transform(X_test)

    # 训练svc
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    print("SVC 准确率 = ",round(svc.score(X_test,y_test),4))
    save = False
    if save:
        dist_pickle = {

            # 分类器
            "svc": svc,
            # 标准器
            "scaler": X_scaler,

            # hog特征参数
            "orient": orient,
            "pix_per_cell": pix_per_cell,
            "cell_per_block": cell_per_block,

            # 空间特征参数
            "spatial_size": spatial_size,

            # 颜色特征参数
            "hist_bins": hist_bins,
        }

        with open("./output_images/svc_pickle.p", "wb") as f:
            pickle.dump(dist_pickle, f)
