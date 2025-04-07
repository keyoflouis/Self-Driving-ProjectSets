import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# 加载图像
binary_warped = mpimg.imread("./IGNORE/warped-example.jpg")


def find_lane_pixels(binary_warped):
    # 计算图像下半部分的直方图
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # 创建一个输出图像，用于绘制结果并可视化
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # 找到直方图左右两部分的峰值
    # 这些峰值将作为左右车道线的起始点
    midpoint = np.int64(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 超参数设置
    # 滑动窗口的数量
    nwindows = 9
    # 窗口宽度（左右边界的偏移量）
    margin = 100
    # 每个窗口中需要重新定位窗口的最小像素数
    minpix = 50

    # 计算窗口高度（基于图像高度和窗口数量）
    window_height = np.int64(binary_warped.shape[0] // nwindows)

    # 找到图像中所有非零像素的坐标
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 初始化左右车道线的当前位置
    leftx_current = leftx_base
    rightx_current = rightx_base

    # 创建空列表，用于存储左右车道线的像素索引
    left_lane_inds = []
    right_lane_inds = []

    # 逐个处理滑动窗口
    for window in range(nwindows):
        # 确定当前窗口的边界（x 和 y 方向）
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        # （左右）车道线检测窗口的左边界x和右边界x
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 在可视化图像上绘制窗口
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # 找到当前窗口内的非零像素
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # 将这些索引添加到列表中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 如果窗口内像素数量超过阈值，则重新定位窗口中心
        if len(good_left_inds) > minpix:
            leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

    # 将索引列表合并为数组
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # 如果没有找到足够的像素，避免报错
        pass

    # 提取左右车道线的像素位置,(left_lane_inds存储的是nonzero数组的索引)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # 首先找到车道线像素
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # 使用二次多项式拟合左右车道线
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 生成用于绘制的 x 和 y 坐标
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # 如果拟合失败，输出默认值
        print('函数未能成功拟合车道线！')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # 可视化部分
    # 标记左右车道线区域
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # 绘制多项式曲线
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


# 执行车道线检测和拟合
out_img = fit_polynomial(binary_warped)

# 显示结果
plt.imshow(out_img)
plt.show()