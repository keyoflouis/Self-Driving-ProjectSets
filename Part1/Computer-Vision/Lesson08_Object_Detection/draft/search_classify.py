import os
import time
import matplotlib.image as mpimg
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

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
        return features ,hog_image

    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       block_norm="L2-Hys",
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualize=vis,
                       feature_vector=feature_vec)

        return features


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    输入图片路径列表，转换为 color_space

    :param imgs:
    :param color_space:

    :param spatial_feat:
    :param spatial_size:

    :param hist_feat:
    :param hist_bins:

    :param hog_feat:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:

    :return:
    '''

    features = []

    for file in imgs:
        file_features = []
        image = mpimg.imread(file)

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

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
            if hog_channel == "ALL":
                hog_feature = []
                for channel in range(feature_image.shape[2]):
                    hog_feature.extend(
                        get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block))
                hog_feature = np.ravel(hog_feature)
            else:
                hog_feature = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block)

            file_features.append(hog_feature)

        features.append(np.concatenate(file_features))

    return features

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
        img_features.append(spatial_features)

    # 5) 如果设置了标志，计算直方图特征
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
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
        img_features.append(hog_features)

    # 9) 返回连接后的特征数组
    return np.concatenate(img_features)

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # 被搜索区域
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # 计算x/y方向步长
    nx_pix_per_step  = int(xy_window[0]*(1-xy_overlap[0]))
    ny_pix_per_step = int(xy_window[1]*(1-xy_overlap[1]))
    
    # 计算x/y方向有多少个窗口
    nx_buffer = int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = int(xy_window[1] * (xy_overlap[1]))
    nx_windows = int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = int((yspan - ny_buffer) / ny_pix_per_step)
    
    # 存储窗口位置的window_list
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # 计算窗口的位置
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # 添加到列表中
            window_list.append(((startx, starty), (endx, endy)))    

    return window_list

def draw_boxes(img,bboxes,color=(0,0,255),thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy,bbox[0],bbox[1],color,thick)

    return imcopy

def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2,hog_channel=0,
                  spatial_feat=True, hist_feat=True, hog_feat=True):
    ''' 传入图片，寻找车辆所在的窗口 '''

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

        print(features.shape)

        # 5) 缩放提取的特征以输入分类器
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) 使用分类器进行预测
        prediction = clf.predict(test_features)
        # 7) 如果预测为正（prediction == 1），则保存该窗口
        if prediction == 1:
            on_windows.append(window)
    return on_windows



if __name__ == "__main__":
    target_dirs = {'non-vehicles_smallset', 'vehicles_smallset'}

    images = []

    for dir_name in target_dirs:
        start_path = os.path.join('../IGNORE', dir_name)

        for root, _, files in os.walk(start_path):  # 当前目录（字符串），当前目录的子目录列表，当前目录的文件列表
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

    sample_size = 500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

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

    car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                                    hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block, hog_channel=hog_channel,
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    notcars_features = extract_features(notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                                        cell_per_block,
                                        hog_channel, spatial_feat, hist_feat, hog_feat)

    X = np.vstack((car_features, notcars_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcars_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # 标准化
    X_scaler = StandardScaler().fit(X_train)
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

        with open("../IGNORE/svc_pickle.p", "wb") as f:
            pickle.dump(dist_pickle, f)

    image = mpimg.imread('../IGNORE/bbox-example-image.jpg')
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