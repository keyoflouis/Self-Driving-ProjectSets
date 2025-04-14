import numpy as np
import cv2
from matplotlib import pyplot as plt

print_img = True


def lane_historgram(img):
    the_sumArr = np.sum(img[:img.shape[0], :img.shape[1]], axis=0)
    return the_sumArr


def find_window_centroids(image, window_width, window_height, margin, l_center=None, r_center=None, thresh=20):
    ''' 找到左右车道线的中心点 '''

    window_centroids = []
    window = np.ones(window_width)
    offset = int(window_width / 2)

    # 定义底部的高，中间部分的x
    piece_top_y = int(3 * image.shape[0] / 4)
    piece_mid_x = int(image.shape[1] / 2)

    # 使用上一帧拟合的多项式来计算得到l_center和r_center，如果没有上一帧，或中心点附近的像素很少，则重新计算center
    if (((l_center is None) or (r_center is None))
            or (np.sum(image[piece_top_y:, (l_center - margin):(l_center + margin)]) < thresh)
            or (np.sum(image[piece_top_y:, (r_center - margin):(r_center + margin)]) < thresh)):
        # 寻找图片底部左半部分的车道线中心点
        l_sum = lane_historgram(image[piece_top_y:, :piece_mid_x])
        conv_l = np.convolve(window, l_sum)
        l_center = np.argmax(conv_l) - offset

        # 寻找图片底部右半部分的车道线中心点
        r_sum = lane_historgram(image[piece_top_y:, piece_mid_x:])
        conv_r = np.convolve(window, r_sum)
        r_center = np.argmax(conv_r) - offset + piece_mid_x

    # 添加中心到列表，并基于上一次的车道线中心进行循环迭代
    window_centroids.append((l_center, r_center))
    for level in range(1, int(image.shape[0] / window_height)):

        before_l_center = l_center
        before_r_center = r_center

        # 窗口的顶部，底部，以及中间点
        piece_top_y = int(image.shape[0] - (level + 1) * window_height)
        piece_btm_y = int(image.shape[0] - level * window_height)
        piece_mid_x = int(image.shape[1] / 2)

        image_layer = lane_historgram(image[piece_top_y:piece_btm_y, :])
        conv_signal = np.convolve(window, image_layer)

        # 重新获取左右车道线的中心
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

        if ((np.sum(image[piece_top_y:, (l_center - margin):(l_center + margin)]) < thresh)
                or (np.sum(image[piece_top_y:, (r_center - margin):(r_center + margin)]) < thresh)):

            # 重新卷积计算
            l_sum = lane_historgram(image[piece_top_y:piece_btm_y, :piece_mid_x])
            conv_l = np.convolve(window, l_sum)
            l_center = np.argmax(conv_l) - offset

            r_sum = lane_historgram(image[piece_top_y:piece_btm_y, piece_mid_x:])
            conv_r = np.convolve(window, r_sum)
            r_center = np.argmax(conv_r) - offset + piece_mid_x

            if (((np.sum(image[piece_top_y:, (l_center - margin):(l_center + margin)]) < thresh)
                 or (np.sum(image[piece_top_y:, (r_center - margin):(r_center + margin)]) < thresh))):
                window_centroids.append((before_l_center, before_r_center))
            else:
                window_centroids.append((l_center, r_center))

        else:
            window_centroids.append((l_center, r_center))

    return window_centroids


def window_mask(width, height, img, center, level, pix_count_thresh=10):
    '''返回生成的掩膜,设立窗口检测阈值'''
    output = np.zeros_like(img)

    # 窗口顶，底，左，右
    piece_top = int(img.shape[0] - (level + 1) * height)
    piece_btm = int(img.shape[0] - level * height)
    piece_left = int(center - width / 2)
    piece_right = int(center + width / 2)

    output[piece_top:piece_btm, max(0, piece_left):min(piece_right, img.shape[1])] = 1

    # 暂时作为中心检测阈值的替代
    valid_pixels = np.sum(output & img)
    if valid_pixels <= pix_count_thresh:
        output[:, :] = 0

    return output


def gene_window(window_width, window_height, image, window_centroids):
    if len(window_centroids) > 0:

        # 掩膜上的所有窗口为1
        l_point = np.zeros_like(image)
        r_point = np.zeros_like(image)

        for level in range(0, len(window_centroids)):
            l_mask = window_mask(window_width, window_height, image, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, image, window_centroids[level][1], level)
            l_point[(l_point == 1) | (l_mask == 1)] = 1
            r_point[(r_point == 1) | (r_mask == 1)] = 1

        # 合并左右车道中心点到一张图片中
        template = np.array(l_point + r_point, np.uint8)
        output = (l_point, r_point)

        if print_img == False:
            zero_channel = np.zeros_like(template)  # 创建一个零颜色通道
            template_de = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # 将窗口像素设置为绿色
            warpage = np.dstack((image, image, image)) * 255  # 将原始道路像素转换为3个颜色通道
            output_de = cv2.addWeighted(warpage, 1, template_de, 0.5, 0.0)  # 将原始道路图像与窗口结果叠加
            plt.imshow(output_de)
            plt.show()
    else:
        print("无中心点")

    return output


def find_lane_pixel(image, mask):
    masked_image = np.zeros_like(image)

    masked_image[(image[:, :] == 1) & (mask == 1)] = 1

    if print_img == False:
        f, (img, msk, msk_img) = plt.subplots(1, 3)
        f.tight_layout()
        img.set_title("image")
        img.imshow(image)

        msk.set_title("mask")
        msk.imshow(mask)

        msk_img.imshow(masked_image)

        plt.show()

    y_coords, x_coords = np.nonzero(masked_image)

    return y_coords, x_coords


def fit_lane(y, x, image):
    fit = np.polyfit(y, x, 2)
    return fit


def draw_line(input_img, left_fit, right_fit):
    # 画出曲线
    process_frame = (input_img * 255).astype(np.uint8)
    out_put_frame = cv2.cvtColor(process_frame, cv2.COLOR_GRAY2BGR)

    # 生成拟合曲线坐标点
    ploty = np.linspace(0, input_img.shape[0] - 1, input_img.shape[0])
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # 限制坐标范围并转换为整数
        left_fitx = np.clip(left_fitx, 0, input_img.shape[1] - 1).astype(int)
        right_fitx = np.clip(right_fitx, 0, input_img.shape[1] - 1).astype(int)
        ploty = ploty.astype(int)

        # 创建点坐标数组
        left_points = np.array([list(zip(left_fitx, ploty))], dtype=np.int32)
        right_points = np.array([list(zip(right_fitx, ploty))], dtype=np.int32)

        # 绘制曲线（红色左车道，绿色右车道）
        cv2.polylines(out_put_frame, left_points, isClosed=False, color=(0, 0, 255), thickness=5)
        cv2.polylines(out_put_frame, right_points, isClosed=False, color=(0, 255, 0), thickness=5)

    return out_put_frame


def find_lane_pipe(img, left_fit=None, right_fit=None):
    window_width = 50
    window_height = 80
    margin = 100

    if (left_fit is None) or (right_fit is None):
        window_centroids = find_window_centroids(img, window_width, window_height, margin)
    else:
        l_center = int(left_fit[0] * (img.shape[0] ** 2) + left_fit[1] * img.shape[0] + left_fit[2])
        r_center = int(right_fit[0] * (img.shape[0] ** 2) + right_fit[1] * img.shape[0] + right_fit[2])

        # 找到车道线中心点
        window_centroids = find_window_centroids(img, window_width, window_height, margin, l_center, r_center)

    # 生成窗口掩膜图层
    (left_mask, right_mask) = gene_window(window_width, window_height, img, window_centroids)

    # 找到左右车道线的像素
    left_y, left_x = find_lane_pixel(img, left_mask)
    right_y, right_x = find_lane_pixel(img, right_mask)

    # 拟合得到的多项式参数
    left_fit = fit_lane(left_y, left_x, img)
    right_fit = fit_lane(right_y, right_x, img)

    out_put_frame = draw_line(img, left_fit, right_fit)

    return out_put_frame, (left_fit, right_fit)


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    left_fit = None
    right_fit = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = calibrate(frame)
        binary = binary_process_pipeline(frame)
        warped = warper(binary)
        process_frame, (left_fit, right_fit) = find_lane_pipe(warped, left_fit, right_fit)

        out_put_frame = restore_perspective(process_frame)
        out.write(out_put_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from calibration import calibrate
    from binary_image import binary_process_pipeline, print_img
    from perspective_transform import warper, restore_perspective
    import glob

    # # test1 ,test4
    # path = "IGNORE/test_images/*.jpg"
    #
    # images = glob.glob(path)
    #
    # for img_path in images:
    #     # 处理图片
    #     img = cv2.imread(img_path)
    #     cal_img = calibrate(img)
    #     binary = binary_process_pipeline(cal_img)
    #     warped = warper(binary)
    #
    #     out, (left_fit, right_fit) = find_lane_pipe(warped)
    #
    #     out = restore_perspective(out)
    #     cv2.imshow(img_path, out)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    path = "IGNORE/project_video.mp4"
    output_path = "./gene.mp4"

    process_video(path, output_path)
