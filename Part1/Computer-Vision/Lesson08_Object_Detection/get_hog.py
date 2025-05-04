# 导入必要的库
import matplotlib.image as mpimg  # 用于读取图像
import matplotlib.pyplot as plt  # 用于绘图展示
import numpy as np  # 数值计算库
import cv2  # OpenCV库，用于图像处理
import glob  # 用于文件路径匹配
from skimage.feature import hog  # 从scikit-image中导入HOG特征提取功能

# 读取所有车辆图片（JPEG格式）
car_images = glob.glob('./IGNORE/cutout*.jpg')

# 定义函数：返回HOG特征及可视化结果
# 返回值说明：
# - 特征始终是返回的第一个元素
# - 当visualize=True时，图像数据会作为第二个元素返回
# - 否则没有第二个返回值
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):
    """
    该函数接收参数并返回HOG特征（可选展平形式）及可选的视觉化矩阵。
    返回值的第一项始终是特征（如果feature_vector=True则为展平后的特征）。
    当visualize=True时，第二项返回值为视觉化矩阵。
    """

    # p20以及README中有详细讲解
    return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block),
                      block_norm='L2-Hys', transform_sqrt=False,
                      visualize=vis, feature_vector=feature_vec)

    # 显式命名返回值
    hog_features = return_list[0]
    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features


# 随机选择一个车辆图片索引
ind = np.random.randint(0, len(car_images))
# 读取选中的图片
image = mpimg.imread(car_images[ind])
# 将图片转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 调用函数并设置vis=True以获取可视化输出
features, hog_image = get_hog_features(gray, orient=9, pix_per_cell=8,cell_per_block=2, vis=True, feature_vec=False)

# 绘制对比图
fig = plt.figure()
plt.subplot(121)  # 左子图：原始图像
plt.imshow(image, cmap='gray')
plt.title('example')
plt.subplot(122)  # 右子图：HOG可视化
plt.imshow(hog_image, cmap='gray')
plt.title('HOG feature vis')
plt.show()