from apply_sobel import abs_sobel_thresh
from dir_thresh import dir_threshold
from mag_dir import mag_thresh

import matplotlib.image as mping
import matplotlib.pyplot as plt
import numpy as np

image = mping.imread("IGNORE/signs_vehicles_xygrad.png")

ksize = 3

gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thres=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

combined = np.zeros_like(dir_binary)
combined[((gradx==1)&(grady==1))|((mag_binary ==1)&(dir_binary==1))]=1

plt.imshow(combined)
plt.show()