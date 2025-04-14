import numpy as np
import matplotlib.pyplot as plt
import cv2


def warper(img):
    '''输入图片，返回透视变换后的图片'''

    # 图片numpy数组的shape是 （高，宽，通道），OpenCV的坐标体系是（宽，高，通道）
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    # 定义逆时针时4点
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # 恢复原尺寸

    return warped

def restore_perspective(img):
    '''返回还原的图片'''
    # 图片numpy数组的shape是 （高，宽，通道），OpenCV的坐标体系是（宽，高，通道）
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    # 定义逆时针时4点
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # 恢复原尺寸

    return warped


if __name__ == "__main__":
    from calibration import calibrate
    from binary_image import binary_process_pipeline
    from matplotlib.patches import Polygon

    path = "IGNORE/test_images/test3.jpg"
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = calibrate(img)
    binary = binary_process_pipeline(img)
    warp = warper(img)

    f, (pic_1, pic_2) = plt.subplots(1, 2)
    f.tight_layout()
    pic_1.set_title("raw")
    pic_2.set_title("warped")

    pic_1.imshow(binary)
    pic_2.imshow(warp)

    plt.show()
