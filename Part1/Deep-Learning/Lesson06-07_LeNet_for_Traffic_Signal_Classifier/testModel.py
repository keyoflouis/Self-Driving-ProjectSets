import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image



import os

model = tf.keras.models.load_model("./model_data")

test_data = []
for filename in os.listdir("./test_img"):
    print(filename)
    file_path = os.path.join("./test_img",filename)
    img = Image.open(file_path)

    img_array = np.array(img)/255.0
    test_data.append(img_array)



for i in test_data:
    result = model.predict(np.expand_dims(i, axis=0))
    class_indices = np.argmax(result, axis=1).astype(int)
    print(class_indices)
