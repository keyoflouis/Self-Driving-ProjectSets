import cv2
import numpy as np

if __name__ == "__main__":
    from binary_image import binary_process_pipeline
    from calibration import calibrate
    from perspective_transform import warper

    # 视频输入输出路径
    input_path = "IGNORE/project_video.mp4"
    output_path = "./output_project_video.mp4"

    # 初始化视频捕捉
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        exit()

    # 视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频写入（使用三通道格式）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换颜色空间为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 应用校准
        calibrated = calibrate(frame_rgb)

        # 处理帧
        binary = binary_process_pipeline(calibrated)

        # 透视变换
        warped = warper(binary)

        # 转换为三通道输出
        binary_uint8 = (warped * 255).astype(np.uint8)
        output_frame = cv2.cvtColor(binary_uint8, cv2.COLOR_GRAY2BGR)

        # 写入帧
        out.write(output_frame)

    # 释放资源
    cap.release()
    out.release()
    print("视频处理完成，输出已保存至:", output_path)
