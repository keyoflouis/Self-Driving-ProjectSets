import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mtimg
import IPython.display
import os

from moviepy import VideoFileClip,  ImageSequenceClip
from rich.jupyter import display


#def process_image(img):
#    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def process_image(frame):
    # 确保输入帧是 RGB 格式
    if frame.ndim == 3 and frame.shape[2] == 3:
        # 将帧转换为灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


        # 将灰度图转换为三通道 RGB
        return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError("输入帧格式不正确")




if __name__ == "__main__":
    test_image_list = os.listdir("test_images/")

    #    print(dir(VideoFileClip))

    # 处理图片并存取输出图片
    for imageFile in test_image_list:
        current_left_line = [0, 0, 0, 0]
        current_right_line = [0, 0, 0, 0]

        img = mtimg.imread("test_images/" + imageFile)
        image_output = process_image(img)
        #mpimg.imsave("test_images_output/" + imageFile, image_output)

    current_left_line = [0, 0, 0, 0]
    current_right_line = [0, 0, 0, 0]
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    clip1 = VideoFileClip('test_videos/solidWhiteRight.mp4')

    processed_frames = [process_image(frame) for frame in clip1.iter_frames(fps=clip1.fps, dtype='uint8')]
    white_clip = ImageSequenceClip(processed_frames, fps=clip1.fps)
    white_clip.write_videofile(white_output, fps=clip1.fps, audio=False)