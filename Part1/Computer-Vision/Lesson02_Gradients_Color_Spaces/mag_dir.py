import numpy as np
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt

def mag_thresh(img, sobel_kernel=3, mag_thres=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=sobel_kernel)

    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    abs_mag = np.uint8(magnitude * 255 / np.max(magnitude))

    mask = np.zeros_like(abs_mag)

    mask[(abs_mag >= mag_thres[0]) & (abs_mag <= mag_thres[1])] = 1

    return mask

if __name__ =="__main__":
    image = mping.imread('./IGNORE/signs_vehicles_xygrad.png')
    grad_binary = mag_thresh(image, sobel_kernel=9, mag_thres=(30, 100))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax2.imshow(grad_binary)

    ax1.set_title('Original Image', fontsize=50)
    ax2.set_title('Threshold Image', fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
