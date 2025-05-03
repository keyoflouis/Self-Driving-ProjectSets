import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

# from skimage.feature import hog
# from skimage import color, exposure
# 图像分为车辆和非车辆

images = glob.glob('IGNORE/*.jpg')
print(images)
cars = []
notcars = []

for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)


# 定义一个函数来返回数据集的一些特征
def data_look(car_list, notcar_list):
    data_dict = {}
    data_dict["n_cars"] = len(car_list)
    data_dict["n_notcars"] = len(notcar_list)
    example_img = mpimg.imread(car_list[0])
    data_dict["image_shape"] = example_img.shape
    data_dict["data_type"] = example_img.dtype
    return data_dict


data_info = data_look(cars, notcars)

print('你的函数返回了',
      data_info["n_cars"], '辆汽车和',
      data_info["n_notcars"], '辆非汽车')
print('大小为: ', data_info["image_shape"], ' 和数据类型:',
      data_info["data_type"])
# 为了好玩，随机选择汽车/非汽车索引并绘制示例图像
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# 读取汽车/非汽车图像
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# 绘制示例图像
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('示例汽车图像')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('示例非汽车图像')
