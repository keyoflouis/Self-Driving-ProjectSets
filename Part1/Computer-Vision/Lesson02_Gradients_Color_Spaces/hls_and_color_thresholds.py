import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mping

image = mping.imread("IGNORE/test1.jpg")

print(image.shape)

R = image[:, :, 0]
G = image[:, :, 1]
B = image[:, :, 2]

plt.imshow(R, cmap='gray')
plt.show()
plt.imshow(G, cmap='gray')
plt.show()

plt.imshow(B, cmap='gray')
plt.show()

