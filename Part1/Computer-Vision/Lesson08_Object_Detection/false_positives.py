import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label

# 读取保存了边界框的pickle文件
# "all_bboxes"列表中的每个元素包含
# 对应上述某张图片的边界框列表
box_list = pickle.load(open("IGNORE/bbox_pickle.p", "rb"))

print(box_list)

# 读取与上图类似的测试图像
image = mpimg.imread('IGNORE/test_image.jpg')
heat = np.zeros_like(image[:, :, 0]).astype(float)


def add_heat(heatmap, bbox_list):
    # 遍历边界框列表
    for box in bbox_list:
        # 对每个边界框内的所有像素点加1
        # 假设每个"box"的格式为((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # 返回更新后的热力图
    return heatmap  # 遍历边界框列表


def apply_threshold(heatmap, threshold):
    # 将低于阈值的像素清零
    heatmap[heatmap <= threshold] = 0
    # 返回阈值处理后的热力图
    return heatmap


def draw_labeled_bboxes(img, labels):
    # 遍历所有检测到的车辆
    for car_number in range(1, labels[1] + 1):
        # 找到当前车辆编号对应的所有像素点
        nonzero = (labels[0] == car_number).nonzero()
        # 获取这些像素点的x和y坐标
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # 根据x和y的最小/最大值确定边界框
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # 在图像上绘制边界框
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # 返回绘制后的图像
    return img


# 为边界框列表中的每个框添加热力值
heat = add_heat(heat, box_list)

# 应用阈值处理以帮助消除误检
heat = apply_threshold(heat, 1)

# 显示时对热力图进行可视化处理
heatmap = np.clip(heat, 0, 255)

# 使用label函数从热力图中找出最终边界框
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('车辆位置')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('热力图')
fig.tight_layout()
plt.show()