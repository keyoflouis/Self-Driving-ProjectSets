import matplotlib.pyplot as plt
import matplotlib.image  as mtimg
import numpy as np
from IPython.core.pylabtools import figsize

img = mtimg.imread("../IGNORE/bbox-example-image.jpg")

f,(win1 ,win2)= plt.subplots(1,2)

win1.imshow(img)

win2.imshow(img.astype(np.float32)/255)

plt.show()
