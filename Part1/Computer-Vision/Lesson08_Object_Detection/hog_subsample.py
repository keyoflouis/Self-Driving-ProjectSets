import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *

# # 添加模块重定向解决版本兼容问题
# import sys
# import sklearn.svm
# import sklearn.preprocessing  # 新增导入
#
# # 模块重定向
# sys.modules['sklearn.svm.classes'] = sklearn.svm
# sys.modules['sklearn.preprocessing.data'] = sklearn.preprocessing  # 新增重定向

# 从序列化的pickle文件中加载预训练的SVC模型
dist_pickle = pickle.load(open("./IGNORE/svc_pickle.p", "rb"))

# 获取SVC对象的属性
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

print(f'orient:{orient} ,'
      f'pix_per_cell: {pix_per_cell}, '
      f'cell_per_blcok: {cell_per_block} ,'
      f'spatial_size: {spatial_size} ,'
      f'hist_bins: {hist_bins} ')

img = mpimg.imread('./IGNORE/bbox-example-image.jpg')


# 定义一个集成函数：通过HOG子采样提取特征并进行预测
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    # 获取图片，提取感兴趣的位置，转换颜色通道
    draw_img = np.copy(img)
    #img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2LUV')

    # 缩放图片
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (int(imshape[1] / scale), int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # 计算可用区块的数量 = (总单元格数 - cell_per_block) // cells_per_step + 1
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 计算一个块内Hog特征数量。
    nfeat_per_block = orient * cell_per_block ** 2

    # 一个窗口大小是64*64像素
    window = 64

    # 计算一个窗口内会有多少个block被覆盖
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1

    # 窗口前进步长
    cells_per_step = 2

    # xy方向的可用窗口数量
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # 计算整张图像的单通道HOG特征，shape(块的行数, 块的列数, 块内单元格行数, 块内单元格列数, 方向数)
    hog1 ,vis_hog1= get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False , vis=True)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    print(hog1.shape,vis_hog1.shape)

    # f, (win1, win2) = plt.subplots(1, 2)
    # f.tight_layout()
    # win1.imshow(vis_hog1,cmap="gray")
    # win2.imshow(ctrans_tosearch)
    # plt.show()

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # 提取当前窗口对应位置的HOG特征，以块为索引
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # 截取当前窗口图像块
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # f ,((win1,win2),(win3,win4) )= plt.subplots(2,2,figsize=(19.2, 10.8))
            # f.tight_layout()
            # win1.imshow(vis_hog1[ytop:ytop + window, xleft:xleft + window])
            # win2.imshow(subimg)
            # win3.imshow(vis_hog1)
            # win4.imshow(ctrans_tosearch)
            # plt.show()


            # 获取空间，颜色特征
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # 特征缩放并预测
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)


            if test_prediction == 1:
                xbox_left = int(xleft * scale)
                ytop_draw = int(ytop * scale)
                win_draw = int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img


ystart = 400
ystop = 656
scale = 1.5

out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                    hist_bins)

plt.imshow(out_img)
plt.show()
