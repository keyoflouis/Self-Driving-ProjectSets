import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

# 读取预处理后的图像
warped = mpimg.imread('./IGNORE/warped-example.jpg')
# 窗口设置
window_width = 50  # 窗口宽度
window_height = 80  # 窗口高度，将图像分为9个垂直层，因为图像高度是720
margin = 100  # 左右滑动搜索的范围


def window_mask(width, height, img_ref, center, level):
    # eg：在第0层（最底层），窗口垂直范围为 720-80 * 1 到 720（即640到720），水平范围为中心±25。

    # 创建一个与图像大小相同的零矩阵
    output = np.zeros_like(img_ref)
    # 在指定的窗口区域内填充1
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):
    # 存储每个层的（左，右）窗口质心位置
    window_centroids = []
    window = np.ones(window_width)

    # 垂直切4片，取底部的一片（靠近车辆）的左半部分求和，找到最密集的列作为左车道起点。
    # 卷积操作增强车道线区域的信号，找到峰值位置。
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)

    # 这里使用的full卷积，卷积核的尺寸在这里是window，默认会在
    temp_l = np.convolve(window, l_sum)
    l_center = np.argmax(temp_l) - window_width / 2

    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    temp_r = np.convolve(window, r_sum)
    r_center = np.argmax(temp_r) - window_width / 2 + int(image.shape[1] / 2)

    # 添加第一层找到的位置
    window_centroids.append((l_center, r_center))

    # 遍历剩余所有层，寻找最大像素位置
    for level in range(1, int(image.shape[0] / window_height)):
        # 对图像的垂直切片进行卷积
        image_layer = np.sum(
            # 窗口顶部：窗口底部
            image[int(image.shape[0] - (level + 1) * window_height): int(image.shape[0] - level * window_height), :],
            axis=0)
        # 卷积
        conv_signal = np.convolve(window, image_layer)

        # 偏移量计算（因为卷积结果对应窗口右侧位置）
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

        # 使用过去的右中心作为参考，找到最佳右质心
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # 添加该层找到的位置
        window_centroids.append((l_center, r_center))

    return window_centroids


# 调用函数找到窗口质心
window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# 如果找到了窗口质心
if len(window_centroids) > 0:
    # 用于绘制所有左右窗口的点
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # 遍历每一层并绘制窗口
    for level in range(0, len(window_centroids)):
        # window_mask是用于绘制窗口区域的函数
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        # 将窗口掩膜中的图形点添加到总像素中
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # 绘制结果
    template = np.array(r_points + l_points, np.uint8)  # 将左右窗口像素合并
    zero_channel = np.zeros_like(template)  # 创建一个零颜色通道
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # 将窗口像素设置为绿色
    warpage = np.dstack((warped, warped, warped)) * 255  # 将原始道路像素转换为3个颜色通道
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # 将原始道路图像与窗口结果叠加
# 如果没有找到窗口质心，只显示原始道路图像
else:
    output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

# 显示最终结果
plt.imshow(output)
plt.title('window fitting results')
plt.show()
