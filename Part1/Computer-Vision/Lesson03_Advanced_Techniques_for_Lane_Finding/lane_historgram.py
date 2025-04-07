import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mping

img = mping.imread("./IGNORE/warped-example.jpg")


def hist(img):
    bottom_half = img[img.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    return histogram


histogram = hist(img)
plt.plot(histogram)
plt.show()
