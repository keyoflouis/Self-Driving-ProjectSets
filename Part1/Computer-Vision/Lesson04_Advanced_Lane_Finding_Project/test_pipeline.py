# import cv2
# import numpy as np
#
# if __name__ == "__main__":
#     from binary_image import binary_process_pipeline
#     from calibration import calibrate
#     from perspective_transform import warper
#     from fit_lane import find_lane_pipe
#
#     # 视频输入输出路径
#     input_path = "IGNORE/project_video.mp4"
#     output_path = "./output_project_video.mp4"
#
#     # 初始化视频捕捉
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         exit()
#
#     # 视频参数
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#
#     # 初始化视频写入（使用三通道格式）
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 转换颜色空间为RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # 应用校准
#         calibrated = calibrate(frame_rgb)
#
#         # 处理帧
#         binary = binary_process_pipeline(calibrated)
#
#         # 透视变换
#         warped = warper(binary)
#
#
#         # 转换为三通道输出
#         binary_uint8 = (warped * 255).astype(np.uint8)
#         output_frame = cv2.cvtColor(binary_uint8, cv2.COLOR_GRAY2BGR)
#
#         # 写入帧
#         out.write(output_frame)
#
#     # 释放资源
#     cap.release()
#     out.release()
#     print("视频处理完成，输出已保存至:", output_path)

import numpy as np
import cv2
from calibration import calibrate
from binary_image import binary_process_pipeline
from perspective_transform import warper
from fit_lane import find_window_centroids, gene_window, find_lane_pixel

# 视频处理参数
INPUT_VIDEO = r"./IGNORE/project_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"


def process_frame(frame):
    """处理单个视频帧的完整流程"""
    # 1. 校准图像
    cal_img = calibrate(frame)

    # 2. 二值化处理
    binary = binary_process_pipeline(cal_img)

    # 3. 透视变换
    warped = warper(binary)

    # 4. 车道线检测（修改后的版本）
    return find_lane_pipe_video(warped)


def find_lane_pipe_video(img):
    """适配视频处理的车道检测函数"""
    # 保持原有逻辑不变，移除可视化相关代码
    window_width = 50
    window_height = 80
    margin = 120

    # 获取窗口中心点
    window_centroids = find_window_centroids(img, window_width, window_height, margin)

    # 生成窗口掩膜
    (left_mask, right_mask) = gene_window(window_width, window_height, img, window_centroids)

    # 获取车道像素点
    left_y, left_x = find_lane_pixel(img, left_mask)
    right_y, right_x = find_lane_pixel(img, right_mask)

    # 创建输出图像
    output = np.zeros_like(img)
    output[left_y, left_x] = 255  # 左车道线设为白色
    output[right_y, right_x] = 255  # 右车道线设为白色

    return output


def process_video():
    # 初始化视频流
    cap = cv2.VideoCapture(INPUT_VIDEO)

    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 处理当前帧
        processed_frame = process_frame(frame)

        # 转换颜色空间用于写入视频
        output_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        # 写入输出视频
        out.write(output_frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()