import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from train import *


class pickle_data:
    def __init__(self, path):
        self.p_file = pickle.load(open(path, "rb"))
        self.svc = self.p_file["svc"]
        self.X_scaler = self.p_file["scaler"]
        self.orient = self.p_file["orient"]
        self.pix_per_cell = self.p_file["pix_per_cell"]
        self.cell_per_block = self.p_file["cell_per_block"]
        self.spatial_size = self.p_file["spatial_size"]
        self.hist_bins = self.p_file["hist_bins"]


def convert_color(img, conv="RGB2YCrCb"):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return

def draw_boxes(img,bboxes,color =(0,0,255),thick=6):
    imgcopy=np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imgcopy,bbox[0],bbox[1],color,thick)

    return imgcopy

def find_cars_heatmap(img, ystart, ystop, scale, p_data: pickle_data):
    ''' 返回车辆热力图 '''

    # 热力图
    draw_image = np.zeros_like(img[:, :, 0])
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (int(imshape[1] / scale), int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # 计算xy方向上可用的块的数量
    nxblocks = (ch1.shape[1] // p_data.pix_per_cell) - p_data.cell_per_block + 1
    nyblocks = (ch2.shape[0] // p_data.pix_per_cell) - p_data.cell_per_block + 1

    # 计算一个块内的hog特证数
    nfeat_per_block = p_data.orient * p_data.cell_per_block ** 2

    window = 64

    # 计算一个窗口内有多少个block会被覆盖
    nblocks_per_window = (window // p_data.pix_per_cell) - p_data.cell_per_block + 1

    # 窗口前进步长
    cells_per_step = 2

    # xy方向上可用窗口数量
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # 求目标范围内hog特征
    hog1 = get_hog_features(ch1, p_data.orient, p_data.pix_per_cell, p_data.cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, p_data.orient, p_data.pix_per_cell, p_data.cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, p_data.orient, p_data.pix_per_cell, p_data.cell_per_block, feature_vec=False)

    pic_rec = np.copy(img)
    single_point=[]

    # 滑动窗口寻找车辆并绘制到热力图
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # 提取结构化hog数组中，窗口内的对应特征信息
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # 提取窗口内的图像
            xleft = xpos * p_data.pix_per_cell
            ytop = ypos * p_data.pix_per_cell
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # debug
            xbox_left = int(xleft * scale)
            ytop_draw = int(ytop * scale)
            win_draw = int(window * scale)
            pic_point.append(((xbox_left,ytop_draw + ystart),(xbox_left + win_draw,ytop_draw + win_draw + ystart)))
            single_point.append(((xbox_left,ytop_draw + ystart),(xbox_left + win_draw,ytop_draw + win_draw + ystart)))

            # 提取特征
            spatial_featurs = bin_spatial(subimg, size=p_data.spatial_size)
            hist_features = color_hist(subimg, nbins=p_data.hist_bins)

            # 标准化
            test_features = p_data.X_scaler.transform(
                np.hstack((spatial_featurs, hist_features, hog_features)).reshape(1, -1))

            # 预测
            test_prediction = p_data.svc.predict(test_features)

            # 绘制热力图
            if test_prediction == 1:
                xbox_left = int(xleft * scale)
                ytop_draw = int(ytop * scale)
                win_draw = int(window * scale)
                draw_image[ytop_draw + ystart:ytop_draw + win_draw + ystart, xbox_left:xbox_left + win_draw] += 1

    if single_window:
        plt.title(f'scale:{scale}')
        plt.imshow(draw_boxes(pic_rec,single_point[0:1]))
        plt.show()

    if print_heatmap:
        plt.title(f'scale:{scale}')
        plt.imshow(draw_image,cmap="gray")
        plt.show()

    return draw_image


def apply_threshold(heat, threshold):
    ''' 处理热力图为（0 - 255），并用阈值过滤 '''

    heat[heat <= threshold] = 0
    heatmap = np.clip(heat, 0, 255)
    return heatmap


def draw_labeled_bboxes(img, labels):
    ''' 遍历label找到框的左上角和右下角 '''

    for car_number in range(1, labels[1] + 1):
        # 找到当前车辆编号对应的所有像素点
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # 根据x和y的最小/最大值绘制边界框
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img


def image_pip(path, p_data: pickle_data, threshold=1):
    ''' 图像处理管道 '''

    image = mpimg.imread(path)

    ystart = 400
    ystop = 656

    # 找到车辆热力图
    scale = 4
    ystart = 300
    ystop = 700
    img = find_cars_heatmap(image, ystart, ystop, scale, p_data)

    scale = 3.2
    ystart = 350
    ystop = 620
    img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

    scale = 2.75
    ystart = 320
    ystop = 570
    img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

    scale = 2.5
    ystart = 320
    ystop = 540
    img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

    scale = 2
    ystart = 350
    ystop = 520
    img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

    scale = 1.5
    ystart = 400
    ystop = 500
    img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

    if windows_view:
        plt.title(f'ALL')
        plt.imshow(draw_boxes(np.copy(image),pic_point))
        plt.show()


    # scale = 1
    # ystart = 350
    # ystop = 500
    # img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

    # 限制热力图数值
    img = apply_threshold(img, threshold)

    # 标记不同车辆
    labels = label(img)

    # 画图
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

windows_view = False
single_window= False
print_heatmap =False
pic_point = []

if __name__ == "__main__":
    # windows_view = True
    # single_window= True
    # print_heatmap = True

    p_data = pickle_data("output_images/svc_pickle.p")

    start_path = "test_images"

    for root, _, files in os.walk(start_path):
        for file in files:
            path = os.path.join(start_path, file)
            img = image_pip(path, p_data)
            plt.title("resault")
            plt.imshow(img)
            plt.show()

    # resault=image_pip("test_images/test1.jpg",p_data)
    # plt.imshow(resault)
    # plt.show()