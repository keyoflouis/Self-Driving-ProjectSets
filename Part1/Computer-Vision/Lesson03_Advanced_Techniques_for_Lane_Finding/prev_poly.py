import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# 加载我们的图像——这应该是一个自上次以来的新帧！
binary_warped = mpimg.imread('./IGNORE/warped-example.jpg')

# 从上一帧获取多项式拟合值
# 确保从项目的上一步中获取实际值！
left_fit = np.array([2.13935315e-04, -3.77507980e-01, 4.76902175e+02])
right_fit = np.array([4.17622148e-04, -4.93848953e-01, 1.11806170e+03])


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    '''返回拟合的车道线的xy的值 '''
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 生成y值，使用所有y值来生成所有x值，得到图片中的拟合车道线(开始，结束，数组长度（确保每一个像素行都有一个对应的点）)
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped):
    # 选择围绕上一帧多项式搜索的宽度
    margin = 100

    # 获取激活的像素点
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 基于上一帧拟合的车道线附近的位置,来捕获当前帧的车道线像素点在nonzero中的位置
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin))
                    & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin))
                    & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # 提取左右车道线所有像素的xy位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 拟合新的多项式
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    # 可视化
    # 创建一个用于绘制的图像和一个显示选择窗口的图像
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # 用颜色标记左右车道线像素
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # 生成一个用于搜索窗口区域的多边形
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # 在空白图像上绘制车道
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # 在图像上绘制多项式曲线
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    # 可视化步骤结束

    return result


# 将图像通过管道处理
# 注意：在你的项目中，你还需要传入上一帧的拟合值
result = search_around_poly(binary_warped)

# 查看输出结果
plt.imshow(result)
plt.show()
