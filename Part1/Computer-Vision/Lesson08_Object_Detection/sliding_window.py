import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('./IGNORE/test_image.jpg')


# 这是前一个练习中的draw_boxes函数
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # 创建图像的副本
    imcopy = np.copy(img)
    # 遍历所有边界框
    for bbox in bboxes:
        # 根据边界框坐标绘制矩形
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # 返回绘制了边界框的图像副本
    return imcopy


# 定义一个函数，接收图像、
# x和y方向的起始/终止位置、
# 窗口尺寸（x和y方向）、
# 以及重叠比例（x和y方向）
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # 当x y_start_stop没有定义的时候，设置为图片尺寸
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # 计算需要被搜索的区域的尺寸
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # 计算x/y方向步长
    nx_pix_per_step = int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = int(xy_window[1] * (1 - xy_overlap[1]))

    # 计算x/y方向有多少个窗口
    nx_buffer = int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = int(xy_window[1] * (xy_overlap[1]))
    nx_windows = int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = int((yspan - ny_buffer) / ny_pix_per_step)

    # 存储窗口位置的window_list
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
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


windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                       xy_window=(128, 128), xy_overlap=(0.5, 0.5))

window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
plt.imshow(window_img)
plt.show()
