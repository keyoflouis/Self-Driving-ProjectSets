import numpy as np


def generate_data():
    '''
    生成用于计算车道曲率的模拟数据。
    实际项目中请使用车道检测算法的输出替代本函数。
    '''
    np.random.seed(0)  # 固定随机种子以保证结果可复现
    ploty = np.linspace(0, 719, num=720)
    quadratic_coeff = 3e-4

    leftx = np.array([200 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                      for y in ploty])
    rightx = np.array([900 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                       for y in ploty])

    leftx = leftx[::-1]
    rightx = rightx[::-1]

    left_fit = np.polyfit(ploty, leftx, 2)
    right_fit = np.polyfit(ploty, rightx, 2)

    return ploty, left_fit, right_fit


def measure_curvature_pixels():
    '''
    计算像素坐标系下的多项式曲率
    '''
    ploty, left_fit, right_fit = generate_data()

    # 选择图像底部（最大y值）作为曲率计算点
    y_eval = np.max(ploty)

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    left_fit[0] = left_fit[0] * xm_per_pix / (ym_per_pix ** 2)
    left_fit[1] = left_fit[1] * xm_per_pix / ym_per_pix
    left_fit[2] = xm_per_pix * left_fit[2]

    right_fit[0] = right_fit[0] * xm_per_pix / (ym_per_pix ** 2)
    right_fit[1] = right_fit[1] * xm_per_pix / ym_per_pix
    right_fit[2] = xm_per_pix * right_fit[2]

    y_eval = y_eval * ym_per_pix

    ##### 曲率半径计算 #####
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** (3 / 2)) / (2 * left_fit[0])  # 左车道线计算

    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** (3 / 2)) / (2 * right_fit[0])  # 右车道线计算



    return left_curverad, right_curverad


# 计算左右车道线曲率半径
left_curverad, right_curverad = measure_curvature_pixels()
print(left_curverad, right_curverad)
# 像素输出近似值 1625.06 和 1976.30
# 转换为米之后： 533.7525889210922 m 648.1574851434297 m
# 实际输出：2323.146038014052 2002.9893199986213
