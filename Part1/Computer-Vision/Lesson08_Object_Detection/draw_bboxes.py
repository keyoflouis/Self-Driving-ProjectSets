import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('./IGNORE/bbox-example-image.jpg')

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    ''' 返回绘制后的图片 '''

    draw_img = np.copy(img)
    for box in bboxes  :
        cv2.rectangle(draw_img,box[0],box[1],color,thick)
    return draw_img

if __name__ =="__main__":

    # Add bounding boxes in this format, these are just example coordinates.
    bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)),
              ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]

    result = draw_boxes(image, bboxes)
    plt.imshow(result)
    plt.show()