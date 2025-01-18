import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mtimg
import os
import numpy as np
from moviepy import VideoFileClip, ImageSequenceClip



def gaussian_blur(image, kerner_size):
    return cv2.GaussianBlur(image, (kerner_size, kerner_size), 0)


def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)


def region_of_interste(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    maksed_image = cv2.bitwise_and(img, mask)

    return maksed_image

def draw_lines(line_img,lines):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img,(x1,y1),(x2,y2),(255,0,0),5)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img,lines)

    return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., y=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, y)


def process_image(image):
    ysize = image.shape[0]
    xsize = image.shape[1]

    # 高斯模糊和Canny检测
    blur_gray = gaussian_blur(image, 11)
    edges = canny(blur_gray, 50, 150)

    # 区域蒙版
    apex_y = 3 / 5 * ysize
    vertices = np.array([[(0, ysize), (0.4 * xsize, apex_y), (0.6 * xsize, apex_y), (xsize, ysize)]], dtype=np.int32)
    masked_edges = region_of_interste(edges, vertices)

    # 霍夫变换检测直线
    line_image = hough_lines(masked_edges, 3, np.pi / 180, threshold=10, min_line_len=50, max_line_gap=35)

    # 绘制检测结果到原图片中
    result = weighted_img(line_image, image, alpha=0.8, beta=1)

    return result


if __name__ == "__main__":
    test_image_list = os.listdir("test_images/")

    # 处理图片并存取输出图片
    for imageFile in test_image_list:

        img = mtimg.imread("test_images/" + imageFile)
        image_output = process_image(img)

        mpimg.imsave("test_images_out/" + imageFile, image_output)


    white_output = 'test_videos_out/solidWhiteRight.mp4'
    clip1 = VideoFileClip('test_videos/solidWhiteRight.mp4')

    # 将视频转换为clip，再对每一帧进行处理，然后组装回clip
    processed_frames = [process_image(frame) for frame in clip1.iter_frames(fps=clip1.fps, dtype='uint8')]
    white_clip = ImageSequenceClip(processed_frames, fps=clip1.fps)

    #输出到文件夹内
    white_clip.write_videofile(white_output, fps=clip1.fps, audio=False)
