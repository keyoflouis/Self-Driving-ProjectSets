import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# 注意：以下导入仅适用于 scikit-learn 版本 <= 0.17
# 如果你使用的是 scikit-learn >= 0.18，请改用以下导入：
from sklearn.model_selection import train_test_split



# 定义一个函数来计算分箱颜色特征
def bin_spatial(img, size=(32, 32)):
    # 使用 cv2.resize().ravel() 创建特征向量
    features = cv2.resize(img, size).ravel()
    # 返回特征向量
    return features


# 定义一个函数来计算颜色直方图特征
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # 分别计算每个颜色通道的直方图
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # 取np.histogram返回的结果（直方统计结果，边界坐标）的第一个，并且拼接。
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # 返回单个直方图、bin中心点和特征向量
    return hist_features


# 定义一个函数从图像列表中提取特征
# 此函数会调用 bin_spatial() 和 color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # 创建一个列表来存储特征向量
    features = []
    # 遍历图像列表
    for file in imgs:
        # 逐个读取图像
        image = mpimg.imread(file)
        # 如果cspace变量不是 'RGB'，则进行颜色转换
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
        # 调用 color_hist() 获取颜色直方图特征
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

# TODO 调整这些值，观察分类器在不同分箱场景下的表现
spatial = 32
histbin = 96

car_features = extract_features(cars, cspace='RGB', spatial_size=(spatial, spatial),
                                hist_bins=histbin, hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(spatial, spatial),
                                   hist_bins=histbin, hist_range=(0, 256))

# 生成数据集
X = np.vstack((car_features, notcar_features)).astype(np.float64)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# 将数据随机分割为训练集和测试集
# rand_state = np.random.randint(0, 100)
rand_state = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# 仅在训练数据上拟合每列的标准化器
X_scaler = StandardScaler().fit(X_train)

# 对 X_train 和 X_test 应用标准化器
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('使用空间分箱大小:', spatial,'和', histbin, '个直方图 bins')
print('特征向量长度:', len(X_train[0]))
# 使用线性支持向量机
svc = LinearSVC()
# 计算支持向量机的训练时间
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), '秒训练支持向量机...')
# 计算支持向量机的准确率
print('支持向量机的测试准确率 = ', round(svc.score(X_test, y_test), 4))
# 计算单个样本的预测时间
t = time.time()
n_predict = 10
print('我的支持向量机预测结果: ', svc.predict(X_test[0:n_predict]))
print('实际', n_predict, '个标签: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t, 5), '秒预测', n_predict, '个标签')