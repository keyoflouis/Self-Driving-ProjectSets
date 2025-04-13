from IPython.core.pylabtools import figsize

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

print_img = True


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    '''返回x/y方向的梯度'''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1)

    sobel = np.abs(sobel)
    sobel = np.uint8(sobel * 255 / np.max(sobel))

    binary = np.zeros_like(sobel)
    binary[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1

    if print_img == False:
        f, (pic1, pic2) = plt.subplots(1, 2)

        pic1.set_title("raw")
        pic1.imshow(img)

        pic2.set_title(f'after abs_sobel_thresh {orient}')
        pic2.imshow(binary)

        plt.show()

    return np.copy(binary)


def mag_thresh(img, ksize=3, thresh=(0, 255)):
    ''' 计算图像梯度值 '''
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
    '''返回梯度方向夹角'''
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
    def color_and_gradient(img, s_thresh=(170, 255), sx_thresh=(20, 100),s_channel_grad_thresh=(15,100)):
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        l_channel = hls_img[:, :, 1]
        s_channel = hls_img[:, :, 2]

        # l通道检测水平方向梯度
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, dx=1, dy=0)
        abs_sobelx = np.abs(sobelx)
        scaled_sobelx = np.uint8(abs_sobelx * 255 / np.max(abs_sobelx))

        l_channel_in_x_binary = np.zeros_like(scaled_sobelx)
        l_channel_in_x_binary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

        # s通道x方向的梯度
        s_channel_in_x_grad = np.abs(cv2.Sobel(s_channel, cv2.CV_64F, dx=1, dy=0))
        scaled_s_channel_in_x_grad = np.uint8(s_channel_in_x_grad * 255 / np.max(abs_sobelx))
        s_channel_in_x_binary = np.zeros_like(scaled_s_channel_in_x_grad)
        s_channel_in_x_binary[(scaled_s_channel_in_x_grad >= s_channel_grad_thresh[0]) & (scaled_s_channel_in_x_grad <= s_channel_grad_thresh[1])] = 1

        # s通道检测颜色纯度
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # 返回二值图片
        combined = np.zeros_like(s_channel)
        combined[(l_channel_in_x_binary == 1) | (s_binary == 1) | (s_channel_in_x_binary == 1) ] = 1
        # combined = np.dstack((np.zeros_like(l_channel_in_x_binary), l_channel_in_x_binary, s_binary)) * 255

        if print_img == False:
            f, (pic1, pic2, pic3) = plt.subplots(1, 3, figsize=(24, 9))
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

    # 灰度处理
    sobelx = abs_sobel_thresh(np.copy(img), orient='x', thresh=(20, 100))
    sobely = abs_sobel_thresh(np.copy(img), orient="y", thresh=(20, 100))
    dir_binary = dir_thresh(np.copy(img), sobel_size=3, thresh=(0.7, 1.3))
    mag_binary = mag_thresh(np.copy(img), ksize=3, thresh=(30, 100))

    # 纯度处理
    color_binary = color_and_gradient(np.copy(img))

    # 结合
    combined = np.zeros_like(mag_binary)
    combined[(((sobelx == 1) & (sobely == 1)) | ((dir_binary == 1) & (mag_binary == 1)) | (color_binary == 1))] = 1

    if print_img == False:
        f, (pic_1, pic_2) = plt.subplots(1, 2)
        f.tight_layout()
        pic_1.set_title("raw")
        pic_1.imshow(img)

        pic_2.set_title("binary image")
        pic_2.imshow(combined,cmap='gray')
        plt.show()

    return combined


if __name__ == "__main__":
    from calibration import calibrate

    path = "IGNORE/test_images/test1.jpg"

    images = glob.glob(path)

    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = calibrate(img)  # 返回矫正后的RGB图片

        binary_process_pipeline(img)
