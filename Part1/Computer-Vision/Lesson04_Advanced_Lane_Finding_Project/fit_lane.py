import numpy as np
import cv2
import matplotlib.pyplot as plt


def lane_historgram(img):
    the_sumArr = np.sum(img[:img.shape[0], :img.shape[1]], axis=0)
    return the_sumArr


def find_window_centroids(image, window_width, window_height, margin):
    ''' 找到左右车道线的中心点 '''

    window_centroids = []
    window = np.ones(window_width)
    offset = window_width / 2

    # 定义底部的高，中间部分的x
    piece_top_y = int(3 * image.shape[0] / 4)
    piece_mid_x = int(image.shape[1] / 2)

    # 寻找图片底部左半部分的车道线中心点
    l_sum = lane_historgram(image[piece_top_y:, :piece_mid_x])
    conv_l = np.convolve(window, l_sum)
    l_center = np.argmax(conv_l) - offset

    # 寻找图片底部右半部分的车道线中心点
    r_sum = lane_historgram(image[piece_top_y:, piece_mid_x:])
    conv_r = np.convolve(window, r_sum)
    r_center = np.argmax(conv_r) - offset + piece_mid_x

    # 添加中心到列表，并基于上一次的车道线中心进行循环迭代
    window_centroids.append((l_center, r_center))
    for level in range(1, int(image.shape[0] / window_height)):
        # 窗口的顶部，底部，以及中间点
        piece_top_y = int(image.shape[0] - (level + 1) * window_height)
        piece_btm_y = int(image.shape[0] - level * window_height)
        piece_mid_x = int(image.shape[1] / 2)

        image_layer = lane_historgram(image[piece_top_y:piece_btm_y, :])
        conv_signal = np.convolve(window, image_layer)

        # 重新获取左右车道线的中心
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

        window_centroids.append((l_center, r_center))

    return window_centroids


def window_mask(width, height, img, center, level):
    output = np.zeros_like(img)

    # 窗口顶，底，左，右
    piece_top = int(img.shape[0] - (level + 1) * height)
    piece_btm = int(img.shape[0] - level * height)
    piece_left = int(center - width / 2)
    piece_right = int(center + width / 2)

    output[piece_top:piece_btm, max(0, piece_left):min(piece_right, img.shape[1])] = 1
    return output


def get_window_pixel(window_width, window_height, image, window_centroids):
    if len(window_centroids) > 0:

        # 掩膜上的所有窗口为1
        l_point = np.zeros_like(image)
        r_point = np.zeros_like(image)

        for level in range(0, len(window_centroids)):
            l_mask = window_mask(window_width, window_height, image, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, image, window_centroids[level][1], level)
            l_point[(l_point == 255) | (l_mask == 1)] = 255
            r_point[(r_point == 255) | (r_mask == 1)] = 255

        # 合并左右车道中心点到一张图片中
        template = np.array(l_point + r_point, np.uint8)

        zero_channel = np.zeros_like(template)  # 创建一个零颜色通道
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # 将窗口像素设置为绿色
        warpage = np.dstack((warped, warped, warped)) * 255  # 将原始道路像素转换为3个颜色通道
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # 将原始道路图像与窗口结果叠加

    else:
        print("无中心点")

    return output


def fit_lane(img):
    window_width = 50
    window_height = 80
    margin = 100

    window_centroids = find_window_centroids(img, window_width, window_height, margin)

    out = get_window_pixel(window_width, window_height, img, window_centroids)

    plt.imshow(out)
    plt.show()

    return


if __name__ == "__main__":
    from calibration import calibrate
    from binary_image import binary_process_pipeline
    from perspective_transform import warper

    # test1 ,test4
    path = "IGNORE/test_images/test1.jpg"
    img = cv2.imread(path)

    cal_img = calibrate(img)
    binary = binary_process_pipeline(cal_img)
    warped = warper(binary)

    fit_lane(warped)

    pass
