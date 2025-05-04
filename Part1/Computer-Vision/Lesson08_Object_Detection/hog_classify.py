import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# 注意：以下导入仅适用于 scikit-learn 版本 <= 0.17
# 对于 scikit-learn >= 0.18 的版本，请使用：
from sklearn.model_selection import train_test_split



# 定义一个函数来返回HOG特征和可视化结果
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # 如果 vis==True 则返回两个输出
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # 否则返回一个输出
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features


# 定义一个函数从图像列表中提取特征
# 该函数会调用 bin_spatial() 和 color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # 创建一个列表来存储特征向量
    features = []
    # 遍历图像列表
    for file in imgs:
        # 逐个读取图像
        image = mpimg.imread(file)
        # 如果cspace不是'RGB'，则进行颜色转换
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # 调用 get_hog_features()，参数 vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 将新特征向量添加到特征列表中
        features.append(hog_features)
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

# 减少样本数量，因为HOG特征计算较慢
# 测试评估器在CPU时间超过13秒后会超时
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

### TODO：调整这些参数并观察结果变化
colorspace = 'HSV'  # 可选值为 RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # 可选值为 0, 1, 2, 或 "ALL"

t = time.time()
car_features = extract_features(cars, cspace=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                   hog_channel=hog_channel)
t2 = time.time()
print(round(t2 - t, 2), '秒提取HOG特征...')

# 合成数据集
X = np.vstack((car_features, notcar_features)).astype(np.float64)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# 将数据随机分为训练集和测试集
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# 对每列数据进行标准化
X_scaler = StandardScaler().fit(X_train)
# 对X应用标准化
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('使用:', orient, '方向数', pix_per_cell,
      '每个单元的像素数', cell_per_block, '每个块的单元数')
print('特征向量长度:', len(X_train[0]))
# 使用线性SVC
svc = LinearSVC()
# 检查SVC的训练时间
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), '秒训练SVC...')
# 检查SVC的准确率
print('SVC的测试准确率 = ', round(svc.score(X_test, y_test), 4))
# 检查单个样本的预测时间
t = time.time()
n_predict = 10
print('我的SVC预测结果: ', svc.predict(X_test[0:n_predict]))
print('实际标签: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t, 5), '秒预测', n_predict, '个标签')