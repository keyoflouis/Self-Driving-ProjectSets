import cv2
import matplotlib.pyplot as plt

from image_test import *



def process_video(video_path,output_path,p_data,threshold=2):
    ''' 视频处理管道 '''

    cap =cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps =cap.get(cv2.CAP_PROP_FPS)

    fource =cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path,fource,fps,(frame_width,frame_height))
    ystart = 400
    ystop = 656

    while cap.isOpened():
        ret,image  = cap.read()
        if not ret:
            break

        ystart = 400
        ystop = 656

        # 找到车辆热力图
        scale = 4
        ystart = 300
        ystop = 700
        img = find_cars_heatmap(image, ystart, ystop, scale, p_data)

        scale = 3.2
        ystart = 350
        ystop = 620
        img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

        scale = 2.75
        ystart = 320
        ystop = 570
        img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

        scale = 2.5
        ystart = 320
        ystop = 540
        img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

        scale = 2
        ystart = 350
        ystop = 520
        img += find_cars_heatmap(image, ystart, ystop, scale, p_data)

        scale = 1.5
        ystart = 400
        ystop = 500
        img += find_cars_heatmap(image, ystart, ystop, scale, p_data)


        # 限制热力图数值
        img = apply_threshold(img, threshold)

        # plt.imshow(img,cmap="gray")
        # plt.show()


        # 标记不同车辆
        labels = label(img)

        # 画图
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        out.write(draw_img)

    cap.release()
    out.release()

if __name__ == "__main__":


    video_path = "project_video.mp4"
    output_path = "output_images/output_video.mp4"

    p_data = pickle_data("output_images/svc_pickle.p")

    process_video(video_path,output_path,p_data)

