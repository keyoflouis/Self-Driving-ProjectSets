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



def download(url, file):
    """    从 url 下载文件
    :param url: 文件的 URL
    :param file: 本地文件路径
    """
    if not os.path.isfile(file):
        print('正在下载 ' + file + '...')
        urlretrieve(url, file)
        print('下载完成')

# 下载训练和测试数据集
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')

# 确保文件未损坏
assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',\
        'notMNIST_train.zip 文件已损坏。删除该文件后重试。'
assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',\
        'notMNIST_test.zip 文件已损坏。删除该文件后重试。'

# 等待所有文件下载完成
print('所有文件已下载。')