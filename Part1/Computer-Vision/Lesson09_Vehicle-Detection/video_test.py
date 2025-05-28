import cv2

from image_test import *



def process_video(video_path,output_path,p_data,threshold=1):
    ''' 视频处理管道 '''

    cap =cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps =cap.get(cv2.CAP_PROP_FPS)

    fource =cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path,fource,fps,(frame_width,frame_height))


    while cap.isOpened():
        ret,frame  = cap.read()
        if not ret:
            break

        # 找到车辆热力图
        img = find_cars_heatmap(frame, ystart, ystop, scale, p_data)

        # 限制热力图数值
        img = apply_threshold(img, threshold)

        # 标记不同车辆
        labels = label(img)

        # 画图
        draw_img = draw_labeled_bboxes(np.copy(frame), labels)

        out.write(draw_img)

    cap.release()
    out.release()

if __name__ == "__main__":

    ystart = 400
    ystop = 656
    scale = 1.5
    video_path = "test_video.mp4"
    output_path = "output_images/output_video.mp4"

    p_data = pickle_data("output_images/svc_pickle.p")

    process_video(video_path,output_path,p_data)

