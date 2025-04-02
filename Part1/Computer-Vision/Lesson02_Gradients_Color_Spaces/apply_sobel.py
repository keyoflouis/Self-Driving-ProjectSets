import numpy as np
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt

def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1)

    sobel = np.abs(sobel)
    sobel = np.uint8(sobel * 255 / np.max(sobel))

    # masked = [1 if (i > thresh_min and i<thresh_max) else 0 for i in sobel]
    binary_output = np.zeros_like(sobel)

    binary_output[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1

    return np.copy(binary_output)

if __name__ =="__main__":
    image = mping.imread('./IGNORE/signs_vehicles_xygrad.png')

    grad_binary = abs_sobel_thresh(image, orient='y', thresh_min=20, thresh_max=100)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax2.imshow(grad_binary)

    ax1.set_title('Original Image', fontsize=50)
    ax2.set_title('Threshold Image', fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
