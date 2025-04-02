import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mping

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1,ksize=sobel_kernel)

    abs = np.arctan2(np.abs(sobely), np.abs(sobelx))
    binary_out = np.zeros_like(abs)
    binary_out[(abs >= thresh[0]) & (abs <= thresh[1])] = 1

    return binary_out

if __name__ =="__main__":
    image = mping.imread("IGNORE/signs_vehicles_xygrad.png")
    grad_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax2.imshow(grad_binary)

    ax1.set_title('Original Image', fontsize=50)
    ax2.set_title('Threshold Image', fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

