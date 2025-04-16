from IPython.core.pylabtools import figsize

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

print_img = True

# x,y方向上的梯度
print_sobel = False
# 纯度与亮度梯度
print_color_and_grad = False
# 打印结果
print_binary_result = False


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    '''传入RGB 返回x/y方向的梯度'''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1)

    sobel = np.abs(sobel)
    sobel = np.uint8(sobel * 255 / np.max(sobel))

    binary = np.zeros_like(sobel)
    binary[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1

    if print_sobel == True:
        f, (pic1, pic2) = plt.subplots(1, 2)

        pic1.set_title("raw")
        pic1.imshow(img)

        pic2.set_title(f'after abs_sobel_thresh {orient}')
        pic2.imshow(binary)

        plt.show()

    return np.copy(binary)


def mag_thresh(img, ksize=3, thresh=(0, 255)):
    '''传入RGB 计算图像梯度值 '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=ksize)

    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    abs_mag = np.uint8(mag * 255 / np.max(mag))

    binary = np.zeros_like(abs_mag)
    binary[(abs_mag >= thresh[0]) & (abs_mag <= thresh[1])] = 1

    if print_img == False:
        f, (pic1, pic2) = plt.subplots(1, 2)
        pic1.set_title('raw')
        pic1.imshow(img)
        pic2.set_title('after mag_thresh')
        pic2.imshow(binary)
        plt.show()

    return binary


def dir_thresh(img, sobel_size=3, thresh=(0, np.pi / 2)):
    '''传入RGB 返回梯度方向夹角'''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel_sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=sobel_size)
    kernel_sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=sobel_size)

    angle = np.arctan2(np.abs(kernel_sobel_y), np.abs(kernel_sobel_x))

    binary = np.zeros_like(angle)
    binary[(angle >= thresh[0]) & (angle <= thresh[1])] = 1

    if print_img == False:
        f, (pic_1, pic_2) = plt.subplots(1, 2)

        pic_1.set_title('raw')
        pic_1.imshow(img)
        pic_2.set_title('after dir_thresh')
        pic_2.imshow(binary)
        plt.show()

    return binary


use_s_channel_grad = True
if use_s_channel_grad == False:

    def color_and_gradient(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        l_channel = hls_img[:, :, 1]
        s_channel = hls_img[:, :, 2]

        # l通道检测水平方向梯度
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, dx=1, dy=0)
        abs_sobelx = np.abs(sobelx)
        scaled_sobelx = np.uint8(abs_sobelx * 255 / np.max(abs_sobelx))
        sobel_x_binary = np.zeros_like(scaled_sobelx)
        sobel_x_binary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

        # s通道检测颜色纯度
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # 返回二值图片
        combined = np.zeros_like(s_channel)
        combined[(sobel_x_binary == 1) | (s_binary == 1)] = 1

        # combined = np.dstack((np.zeros_like(sobel_x_binary), sobel_x_binary, s_binary)) * 255

        if print_img == False:
            f, (pic1, pic2, pic3) = plt.subplots(1, 3, figsize=(24, 9))
            f.tight_layout()
            pic1.set_title('combined')
            pic1.imshow(combined)
            pic2.set_title('l channel')
            pic2.imshow(sobel_x_binary)
            pic3.set_title('s channel')
            pic3.imshow(s_binary)
            plt.show()

        return combined
else:
    def color_and_gradient(img, s_thresh=(170, 255), sx_thresh=(20, 100), s_channel_grad_thresh=(15, 100)):
        ''' 传入RGB 返回纯度与l，s梯度阈值 '''

        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        h_channel = hls_img[:, :, 0]
        l_channel = hls_img[:, :, 1]
        s_channel = hls_img[:, :, 2]

        l_threshed = np.copy(l_channel)
        l_threshed[l_channel < 200] = 0

        # # 创建子图
        # f, ((pic_1, pic_2), (pic_3, pic_4)) = plt.subplots(2, 2, figsize=(10, 8))
        # f.tight_layout()
        #
        # # 显示H通道（调整至0-255范围）
        # pic_1.set_title("H Channel")
        # pic_1.imshow(h_channel, cmap='gray', vmin=0, vmax=180)  # 显式指定数值范围
        #
        # # 显示L通道
        # pic_2.set_title("L Channel")
        # pic_2.imshow(l_threshed, cmap='gray')
        #
        # # 显示S通道
        # pic_3.set_title("S Channel")
        # pic_3.imshow(s_channel, cmap='gray')
        #
        # # 显示原始RGB图像（或转换HLS回RGB）
        # pic_4.set_title("Original RGB")
        # pic_4.imshow(img)  # 建议显示原始输入
        #
        # plt.show()

        # l通道检测水平方向梯度
        sobelx = cv2.Sobel(l_threshed, cv2.CV_64F, dx=1, dy=0)
        abs_sobelx = np.abs(sobelx)
        scaled_sobelx = np.uint8(abs_sobelx * 255 / np.max(abs_sobelx))

        l_channel_in_x_binary = np.zeros_like(scaled_sobelx)
        l_channel_in_x_binary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

        # s通道x方向的梯度
        s_channel_in_x_grad = np.abs(cv2.Sobel(s_channel, cv2.CV_64F, dx=1, dy=0))
        scaled_s_channel_in_x_grad = np.uint8(s_channel_in_x_grad * 255 / np.max(abs_sobelx))
        s_channel_in_x_binary = np.zeros_like(scaled_s_channel_in_x_grad)
        s_channel_in_x_binary[(scaled_s_channel_in_x_grad >= s_channel_grad_thresh[0]) & (
                scaled_s_channel_in_x_grad <= s_channel_grad_thresh[1])] = 1

        # s通道检测颜色纯度
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # 返回二值图片
        combined = np.zeros_like(s_channel)
        combined[(l_channel_in_x_binary == 1) | (s_binary == 1) | (s_channel_in_x_binary == 1)] = 1
        # combined = np.dstack((np.zeros_like(l_channel_in_x_binary), l_channel_in_x_binary, s_binary)) * 255

        if print_color_and_grad == True:
            f, ((pic1, pic2), (pic3, pic4)) = plt.subplots(2, 2, figsize=(24, 9))
            f.tight_layout()
            pic1.set_title('combined')
            pic1.imshow(combined)
            pic2.set_title('l channel')
            pic2.imshow(l_channel_in_x_binary)
            pic3.set_title('s channel')
            pic3.imshow(s_binary)
            plt.show()

        return combined


def binary_process_pipeline(img):
    '''处理为二值图像'''

    # f, ((pic_1, pic_2), (pic_3, pic_4)) = plt.subplots(2, 2)
    # f.tight_layout()
    # pic_1.set_title("r")
    # pic_1.imshow(img[:, :, 0])
    #
    # pic_2.set_title("g")
    # pic_2.imshow(img[:, :, 1])
    #
    # pic_3.set_title("b")
    # pic_3.imshow(img[:, :, 2])
    #
    # pic_4.set_title("raw")
    # pic_4.imshow(img)
    #
    # plt.show()

    # 灰度处理
    sobelx = abs_sobel_thresh(np.copy(img), orient='x', thresh=(20, 100))
    sobely = abs_sobel_thresh(np.copy(img), orient="y", thresh=(20, 100))
    dir_binary = dir_thresh(np.copy(img), sobel_size=3, thresh=(0.7, 1.3))
    mag_binary = mag_thresh(np.copy(img), ksize=3, thresh=(30, 100))

    # 纯度处理
    color_binary = color_and_gradient(np.copy(img))

    # 结合
    combined = np.zeros_like(mag_binary)

    temp_sobx_soby = np.zeros_like(combined)
    temp_dir_mag = np.zeros_like(combined)

    temp_sobx_soby[(sobelx == 1) & (sobely == 1)] = 1
    temp_dir_mag[(dir_binary == 1) & (mag_binary == 1)] = 1

    combined[(temp_sobx_soby == 1) | (temp_dir_mag == 1) | (color_binary == 1)] = 1

    if print_binary_result == True:
        f, ((pic_1, pic_2), (pic_3, pic_4)) = plt.subplots(2, 2)
        f.tight_layout()
        pic_1.set_title("color_binary")
        pic_1.imshow(color_binary)

        pic_2.set_title("sobx_soby")
        pic_2.imshow(temp_sobx_soby, cmap='gray')

        pic_3.set_title("dir_mag")
        pic_3.imshow(temp_dir_mag, cmap="gray")

        pic_4.set_title("combined")
        pic_4.imshow(combined, cmap="gray")
        plt.show()

    return combined


def gaussian_blur(img, kerner_size=9):
    return cv2.GaussianBlur(img, (kerner_size, kerner_size), 0)


if __name__ == "__main__":
    from calibration import calibrate

    path = "IGNORE/test_images/*.jpg"

    images = glob.glob(path)

    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = calibrate(img)  # 返回矫正后的RGB图片

        img_without_gaus = binary_process_pipeline(img)
        raw_img = gaussian_blur(img)
        img = binary_process_pipeline(img)

        # f, ((pic1, pic2), (pic3, pic4)) = plt.subplots(2, 2)
        #
        # f.tight_layout()
        # pic1.set_title("raw_img")
        # pic1.imshow(raw_img)
        #
        # pic2.set_title("img")
        # pic2.imshow(img, cmap="gray")
        #
        # pic3.set_title("img_without_gaus")
        # pic3.imshow(img_without_gaus, cmap="gray")
        #
        # plt.show()
