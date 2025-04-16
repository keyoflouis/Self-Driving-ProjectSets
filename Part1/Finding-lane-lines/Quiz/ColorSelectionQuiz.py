import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.jpg')

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

thresholds = (
        (image[:, :, 0] < rgb_threshold[0]) |
        (image[:, :, 1] < rgb_threshold[1]) |
        (image[:, :, 2] < rgb_threshold[2])
)

color_select[thresholds] = [0, 0, 0]

f, ((pic_1, pic_2), (pic_3, pic_4)) = plt.subplots(2, 2)
f.tight_layout()
pic_1.set_title("r")
pic_1.imshow(color_select[:,:,0])

pic_2.set_title("g")
pic_2.imshow(color_select[:,:,1])

pic_3.set_title("b")
pic_3.imshow(color_select[:,:,2])

pic_4.set_title("raw")
pic_4.imshow(image)

plt.show()