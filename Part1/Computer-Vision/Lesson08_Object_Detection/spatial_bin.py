import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取图像
# 您也可以读取cutout2、3、4等以查看其他示例
image = mpimg.imread('./IGNORE/cutout1.jpg')


# 定义一个函数来计算颜色直方图特征
# 将color_space标志作为3字母全大写字符串传递，如'HSV'或'LUV'等。
# 请记住，如果您决定在项目后期使用此函数，
# 如果您使用cv2.imread()读取图像，则从BGR颜色开始！
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # 将图像转换为新的颜色空间（如果指定）
    # 使用cv2.resize().ravel()创建特征向量

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)

    features = cv2.resize(feature_image,size).ravel()

    # 返回特征向量
    return features


feature_vec = bin_spatial(image, color_space='RGB', size=(32, 32))

# 绘制特征
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
plt.show()
