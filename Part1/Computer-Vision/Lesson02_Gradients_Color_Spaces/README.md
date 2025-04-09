
# 生成二值图像

apply_sobel.py 对x/y方向求梯度

mag_dir.py 通过`np.sqrt(sobel_x**2 + sobel_y**2)` 计算图像的梯度值

dir_thresh.py 通过反正切得到梯度的方向

combining_Thresholds.py 筛选同时在x和y方向上都有梯度值，或者满足梯度在某个方向，且值位于某个区间的图像

color_and_gradient.py 利用纯度信息辅助提取车道线（太阳下黄色车道线在灰度梯度下几乎不可见），
用l通道筛选边缘，s通道筛选纯度