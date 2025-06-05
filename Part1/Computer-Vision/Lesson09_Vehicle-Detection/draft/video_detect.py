import matplotlib.pyplot as plt

from image_detect import *

class HeatMap():
    def __init__(self):
        # 每 COUNTER 帧重新初始化一次
        self.counter = 0
        # 当前收集的热图
        self.current_heatmap = None
        # 最近 COUNTER 帧的热图
        self.last_heatmap = None

    def self_update(self, heat):
        # 当视频开始时
        if self.last_heatmap is None:
            # 使用第一帧作为起点
            self.current_heatmap = heat
            self.last_heatmap = heat

        # 收集当前 COUNTER 帧
        if self.counter != COUNTER:
            self.counter += 1
            # 在当前热图上添加
            self.current_heatmap += heat
        # 收集完成
        else:
            # 传递最终热图
            self.last_heatmap = self.current_heatmap
            # 开始新的一轮
            self.counter = 0
            self.current_heatmap = heat


def process_image(image):
    # JPG -> PNG
    image = image.astype(np.float32)/255

    # 获取搜索车辆的窗口列表
    window_list = slide_window(image, y_start_stop=[390, 430])
    # 获取活动窗口
    on_window = search_windows(image, window_list, svc, X_scaler)

    # 应用热图以定位车辆
    heat = np.zeros_like(image[:, :, 0]).astype(float)
    heat = add_heat(heat, on_window)

    # 剔除误检
    heatmap.self_update(heat)
    # 使用最近 COUNTER 帧的热图
    heat = apply_threshold(heatmap.last_heatmap, COUNTER*3)

    # 标记连接的活动区域
    labels = label(heat)
    # 使用标签函数从热图中找到最终边界框
    result = draw_labeled_bboxes((image*255).astype(int), labels)

    # 返回结果图像
    return result

if __name__ =="__main__":
    COUNTER = 5
    heatmap = HeatMap()
    project_output = '../output_images/test_video.mp4'
    clip = VideoFileClip("../project_video.mp4")
    processed_frames = []

    for frame in clip.iter_frames():
        processed_frames.append(process_image(frame))

    out_clip = ImageSequenceClip(processed_frames,fps=clip.fps)
    out_clip.write_videofile(project_output)