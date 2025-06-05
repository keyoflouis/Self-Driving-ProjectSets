import matplotlib.pyplot as plt

from test import *
from train import *


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None]):
    # 如果未定义 x 和 / 或 y 的起始 / 停止位置，则设置为图像大小
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # 初始化一个列表以追加窗口位置
    window_list = []
    # 循环查找 x 和 y 窗口位置
    # 车辆在图像底部似乎更大
    window_sizes = np.array([64, 80, 96, 128, 160])
    x_overlaps = np.array([0.333, 0.5, 0.5, 0.667, 0.667])

    # 计算 x 和 y 方向的跨度
    xspan = x_start_stop[1] - x_start_stop[0]  # 常量
    yspan = y_start_stop[1] - y_start_stop[0]  # 常量
    
    nx_buffer = (window_sizes * x_overlaps).astype(int)  # 向量
    
    # 计算滑动步长和窗口数量
    nx_pix_per_steps = (window_sizes*(1 - x_overlaps)).astype(int)  # 向量
    nx_windows = ((xspan - nx_buffer)/nx_pix_per_steps).astype(int)  # 向量
    
    ny_window = len(window_sizes)  # 常量
    ny_pix_per_step = int(yspan / ny_window + 1)  # 常量

    # 循环遍历感兴趣区域
    for ys in range(ny_window):
        # nx 每步的像素数、nx 窗口数量和窗口大小不同
        window_size = window_sizes[ys]
        nx_pix_per_step = nx_pix_per_steps[ys]
        nx_window = nx_windows[ys]
        for xs in range(nx_window):
            # 计算窗口位置
            startx =  x_start_stop[1] - xs*nx_pix_per_step
            endx = startx - window_size
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + window_size

            # 将窗口位置追加到列表中
            window_list.append(((startx, starty), (endx, endy)))
    # 返回窗口列表
    return window_list

def search_windows(image, windows, clf, scaler):
    # 创建一个空列表以接收正检测窗口
    on_windows = []
    global k
    # 遍历列表中的所有窗口
    for window in windows:
        # 从原始图像中提取测试窗口
        test_img = cv2.resize(image[window[0][1]:window[1][1], window[1][0]:window[0][0]], (64, 64))
        # 使用 single_img_features() 提取该窗口的特征
        features = extract_features(test_img)
        # 将提取的特征进行缩放以便输入到分类器中
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 使用分类器进行预测
        prediction = clf.predict(test_features)

        # 如果为正（prediction == 1），则保存该窗口
        if prediction == 1:
            on_windows.append(window)

    # 返回正检测的窗口
    return on_windows

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=5):
    # 复制图像
    imcopy = np.copy(img)
    # 遍历边界框
    for bbox in bboxes:
        # 根据边界框坐标绘制矩形
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # 返回绘制了边界框的图像副本
    return imcopy

def add_heat(heatmap, bbox_list):
    # 遍历边界框列表
    for box in bbox_list:
        # 为每个边界框内的所有像素加 1
        heatmap[box[0][1]:box[1][1], box[1][0]:box[0][0]] += 1

    # 返回更新后的热图
    return heatmap# 遍历边界框列表

def apply_threshold(heatmap, threshold):
    # 将低于阈值的像素置为 0
    heatmap[heatmap <= threshold] = 0
    # 返回阈值化后的热图
    return heatmap

def draw_labeled_bboxes(img, labels):
    # 遍历所有检测到的车辆
    for car_number in range(1, labels[1]+1):
        # 找到每个 car_number 标签值的像素
        nonzero = (labels[0] == car_number).nonzero()
        # 确定这些像素的 x 和 y 值
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # 根据 x 和 y 的最小值和最大值定义边界框
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # 在图像上绘制边界框
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 5)

    # 返回图像
    return img

if __name__ == "__main__":
    test_images = glob.glob("../test_images/test*.jpg")

    fig, ax = plt.subplots(len(test_images), 3, figsize=(20, 40))
    for idx, image_file in enumerate(test_images):
        # 读取图像
        image = mpimg.imread(image_file)
        image = image.astype(np.float32) / 255

        # 应用滑动窗口以搜索车辆
        window_list = slide_window(image, y_start_stop=[390, 430])
        on_window = search_windows(image, window_list, svc, X_scaler)
        box_image = draw_boxes(image, on_window)

        # 应用热图以定位车辆
        heat = np.zeros_like(image[:, :, 0]).astype(float)
        heat = add_heat(heat, on_window)
        heatmap = apply_threshold(heat, 2)

        # 使用标签函数从热图中找到最终边界框
        labels = label(heatmap)
        result = draw_labeled_bboxes(np.copy(image), labels)

        # 显示
        ax[idx, 0].imshow(box_image)
        ax[idx, 1].imshow(heatmap, cmap='gray')
        ax[idx, 2].imshow(result)

    plt.tight_layout()
    plt.show()