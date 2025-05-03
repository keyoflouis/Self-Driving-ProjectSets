import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob


# 定义计算分箱颜色特征的函数
def bin_spatial(img, size=(32, 32)):
    # 使用cv2.resize().ravel()创建特征向量
    features = cv2.resize(img, size).ravel()
    return features


# 定义计算颜色直方图特征的函数
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # 分别计算各颜色通道的直方图
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # 将直方图拼接为单个特征向量
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


# 定义一个函数，用于从图像列表中提取特征
# 该函数会调用 bin_spatial() 和 color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # 创建一个列表用于存储特征向量
    features = []
    # 遍历图像列表
    for file in imgs:
        # 逐个读取图像
        image = mpimg.imread(file)
        # 如果色彩空间不是'RGB'，则进行颜色转换
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(image)
        # 调用 bin_spatial() 获取空间颜色特征
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 调用 color_hist() 获取颜色直方图特征（现在支持色彩空间选项）
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # 将新特征向量添加到特征列表中
        features.append(np.concatenate((spatial_features, hist_features)))
    # 返回特征向量列表
    return features


import os

target_dirs = {'non-vehicles_smallset', 'vehicles_smallset'}
images = []

# 只遍历目标目录（跳过其他目录提升效率）
for dir_name in target_dirs:
    start_path = os.path.join('./IGNORE/', dir_name)
    # 递归遍历当前目标目录
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                images.append(os.path.join(root, file))
cars = []
notcars = []
for image in images:
    if 'notcars' in image:
        notcars.append(image)
    else:
        cars.append(image)

print(len(cars), len(notcars))

car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
                                hist_bins=32, hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
                                   hist_bins=32, hist_range=(0, 256))

if len(car_features) > 0:
    # 堆叠特征向量
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # 拟合每列标准化器
    X_scaler = StandardScaler().fit(X)
    # 应用标准化器
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))
    # 绘制原始特征与标准化特征对比图
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('raw image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('raw feature')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('normalized feature')
    fig.tight_layout()
    plt.show()
else:
    print('您的函数返回了空特征向量...')
