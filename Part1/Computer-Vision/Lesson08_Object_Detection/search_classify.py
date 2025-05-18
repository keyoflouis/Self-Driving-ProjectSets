import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

from lesson_functions import *
# 注意：以下导入仅适用于 scikit-learn 版本 <= 0.17
# 对于 scikit-learn >= 0.18 的版本，请使用：
from sklearn.model_selection import train_test_split
import pickle


# from sklearn.cross_validation import train_test_split


# 定义一个函数，用于从单个图像窗口中提取特征
# 该函数与 extract_features() 非常相似
# 只是针对单个图像而非图像列表
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) 定义一个空列表来接收特征
    img_features = []
    # 2) 如果颜色空间不是 'RGB'，则进行颜色转换
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) 如果设置了标志，计算空间特征
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) 将特征添加到列表中
        img_features.append(spatial_features)
    # 5) 如果设置了标志，计算直方图特征
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) 将特征添加到列表中
        img_features.append(hist_features)
    # 7) 如果设置了标志，计算 HOG 特征
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) 将特征添加到列表中
        img_features.append(hog_features)

    # 9) 返回连接后的特征数组
    return np.concatenate(img_features)


# 定义一个函数，用于传入图像和要搜索的窗口列表（slide_windows() 的输出）
def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) 创建一个空列表来接收检测到的正窗口
    on_windows = []
    # 2) 遍历列表中的所有窗口
    for window in windows:
        # 3) 从原始图像中提取测试窗口
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) 使用 single_img_features() 提取该窗口的特征
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block, hog_channel=hog_channel,
                                       spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        de_fea = features
        # 5) 缩放提取的特征以输入分类器
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) 使用分类器进行预测
        prediction = clf.predict(test_features)
        # 7) 如果预测为正（prediction == 1），则保存该窗口
        if prediction == 1:
            on_windows.append(window)
    # 8) 返回检测到的正窗口
    return on_windows


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

# 减少样本大小，因为
# 测验评估器在 CPU 时间超过 13 秒后会超时
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

### TODO：调整这些参数并观察结果变化。
color_space = 'LUV'  # 可以是 RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG 方向数
pix_per_cell = 8  # HOG 每个单元的像素数
cell_per_block = 2  # HOG 每个块的单元数
hog_channel = "ALL"  # 可以是 0, 1, 2 或 "ALL"
spatial_size = (32, 32)  # 空间分箱维度
hist_bins = 32  # 直方图的分箱数
spatial_feat = True  # 是否启用空间特征
hist_feat = True  # 是否启用直方图特征
hog_feat = True  # 是否启用 HOG 特征
y_start_stop = [400,None]  # 在 slide_window() 中搜索的 y 范围最小值和最大值

car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                   hog_feat=hog_feat)

# 创建一个特征向量的堆叠数组
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# 定义标签向量
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# 将数据分割为随机的训练集和测试集
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# 对每列数据进行标准化
X_scaler = StandardScaler().fit(X_train)
# 对 X 应用标准化
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('使用:', orient, '个方向', pix_per_cell, '像素每单元和', cell_per_block, '单元每块')
print('特征向量长度:', len(X_train[0]))
# 使用线性 SVC
svc = LinearSVC()
# 检查 SVC 的训练时间
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), '秒训练 SVC...')
# 检查 SVC 的准确率
print('SVC 的测试准确率 = ', round(svc.score(X_test, y_test), 4))
# 检查单个样本的预测时间
t = time.time()

save = False

if save:
    dist_pickle ={
        "svc":svc,
        "scaler":X_scaler,
        "orient":orient,
        "pix_per_cell":pix_per_cell,
        "cell_per_block":cell_per_block,
        "spatial_size":spatial_size,
        "hist_bins":hist_bins,
    }

    with open("./IGNORE/svc_pickle.p", "wb") as f:
        pickle.dump(dist_pickle, f)


image = mpimg.imread('./IGNORE/bbox-example-image.jpg')
draw_image = np.copy(image)

# 如果你从 .png 图像（通过 mpimg 缩放到 0 到 1）中提取训练数据，
# 并且你正在搜索的图像是 .jpg（缩放到 0 到 255），
# 请取消以下行的注释
# image = image.astype(np.float32)/255


windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=(96, 96),
                       xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                             spatial_size=spatial_size, hist_bins=hist_bins,
                             orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)
window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

plt.imshow(window_img)
plt.show()

