# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# def plot3d(pixels, colors_rgb,axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
#     """Plot pixels in 3D."""
#
#     # Create figure and 3D axes
#     fig = plt.figure(figsize=(8, 8))
#     ax = Axes3D(fig)
#
#     # Set axis limits
#     ax.set_xlim(*axis_limits[0])
#     ax.set_ylim(*axis_limits[1])
#     ax.set_zlim(*axis_limits[2])
#
#     # Set axis labels and sizes
#     ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
#     ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
#     ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
#     ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)
#
#     # Plot pixel values with colors given in colors_rgb
#     ax.scatter(
#         pixels[:, :, 0].ravel(),
#         pixels[:, :, 1].ravel(),
#         pixels[:, :, 2].ravel(),
#         c=colors_rgb.reshape((-1, 3)), edgecolors='none')
#
#     return ax  # return Axes3D object for further manipulation
#
#
# # Read a color image
# img = cv2.imread("IGNORE/000275.png")
#
# # Select a small fraction of pixels to plot by subsampling it
# scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
# img_small = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
#
# # Convert subsampled image to desired color space(s)
# img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
# img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
# img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
#
# # Plot and show
# ax = plot3d(img_small_RGB, img_small_rgb)
# plt.show()
#
# plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取图像并转换颜色空间
img_bgr = cv2.imread('IGNORE/cutout2.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)

# 下采样设置（减少数据量）
step = 5  # 调整此值控制点的密度（越大点越少）
img_rgb_sampled = img_rgb[::step, ::step]
img_hls_sampled = img_hls[::step, ::step]

# 重塑为像素数组并归一化颜色
pixels_rgb = img_rgb_sampled.reshape(-1, 3)
pixels_hls = img_hls_sampled.reshape(-1, 3)
colors_rgb = pixels_rgb / 255.0  # Matplotlib需要0-1范围的RGB

# 提取RGB通道值
r, g, b = pixels_rgb[:, 0], pixels_rgb[:, 1], pixels_rgb[:, 2]

# 提取HLS通道值
h, l, s = pixels_hls[:, 0], pixels_hls[:, 1], pixels_hls[:, 2]

# 创建3D可视化
fig = plt.figure(figsize=(18, 8))

# RGB颜色空间子图
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(r, g, b, c=colors_rgb, marker='o', s=5, alpha=0.5)
ax1.set_xlabel('Red (0-255)', fontsize=10)
ax1.set_ylabel('Green (0-255)', fontsize=10)
ax1.set_zlabel('Blue (0-255)', fontsize=10)
ax1.set_title('RGB Color Space', fontsize=12)
ax1.set_xlim(0, 255)
ax1.set_ylim(0, 255)
ax1.set_zlim(0, 255)
ax1.view_init(elev=15, azim=30)  # 调整视角

# HLS颜色空间子图
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(h, l, s, c=colors_rgb, marker='o', s=5, alpha=0.5)
ax2.set_xlabel('Hue (0-180)', fontsize=10)
ax2.set_ylabel('Lightness (0-255)', fontsize=10)
ax2.set_zlabel('Saturation (0-255)', fontsize=10)
ax2.set_title('HLS Color Space', fontsize=12)
ax2.set_xlim(0, 180)
ax2.set_ylim(0, 255)
ax2.set_zlim(0, 255)
ax2.view_init(elev=15, azim=30)  # 调整视角

plt.tight_layout()
plt.show()

# import cv2

# def gene_3d_scatter(img_rgb, mode_default=['RGB', 'HLS']):
#     ''' 需要输入RGB图片 '''

#     count_subpic = len(mode_default)

#     if (count_subpic == 0 or count_subpic > 4):
#         return None

#     def _gene_subplot(idx, img_rgb, mode):

#         if mode != 'RGB':
#             if mode == "HLS":
#                 mode = cv2.COLOR_RGB2HLS
#             else:
#                 print(f'没有录入 mode:{mode}')
#                 return

#             img_cvt = cv2.cvtColor(img_rgb, mode)

#             # 下采样设置（减少数据量）
#             step = 5  # 调整此值控制点的密度（越大点越少）
#             img_rgb_sampled = img_cvt[::step, ::step]

#             # 重塑为像素数组并归一化颜色
#             pixels_rgb = img_rgb_sampled.reshape(-1, 3)
#             colors_rgb = pixels_rgb / 255.0  # Matplotlib需要0-1范围的RGB

#             # 提取RGB通道值
#             r, g, b = pixels_rgb[:, 0], pixels_rgb[:, 1], pixels_rgb[:, 2]

#             # RGB颜色空间子图
#             ax1 = fig.add_subplot(idx, projection='3d')
#             scatter1 = ax1.scatter(r, g, b, c=colors_rgb, marker='o', s=5, alpha=0.5)
#             ax1.set_xlabel('Red (0-255)', fontsize=10)
#             ax1.set_ylabel('Green (0-255)', fontsize=10)
#             ax1.set_zlabel('Blue (0-255)', fontsize=10)
#             ax1.set_title('RGB Color Space', fontsize=12)
#             ax1.set_xlim(0, 255)
#             ax1.set_ylim(0, 255)
#             ax1.set_zlim(0, 255)
#             ax1.view_init(elev=15, azim=30)  # 调整视角

#         else:


#         return

#     r_c = "22"
#     fig = plt.figure(figsize=(18, 8))

#     for i,mode in zip(range(count_subpic),mode_default):
#         r_c_i = r_c + f'{i + 1}'


#     pass


# if __name__ == '__main__':
#     pass
