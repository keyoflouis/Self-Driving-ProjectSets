

import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile



def uncompress_features_labels(file):


    features = []
    labels = []

    with ZipFile(file) as zipf:
        # 进度条
        filenames_pbar = tqdm(zipf.namelist(), unit='files')

        # 从所有文件中获取特征和标签
        for filename in filenames_pbar:
            # 检查文件是否为目录
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # 将图像数据加载为一维数组
                    # 我们使用 float32 以节省内存空间
                    feature = np.array(image, dtype=np.float32).flatten()

                # 从文件名中获取字母。这就是图像的字母。
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

# 从 zip 文件中获取特征和标签
train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')

# 限制数据量，以便在 Docker 容器中使用
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)

# 设置特征工程的标志。这将防止你跳过重要步骤。
is_features_normal = False
is_labels_encod = False

# 等待所有特征和标签解压完成
print('所有特征和标签已解压。')