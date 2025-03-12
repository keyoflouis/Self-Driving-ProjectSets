import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd
import imageio.v2 as imageio
from alexnet import AlexNet

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# NOTE: By setting `feature_extract` to `True` we return
# the second to last layer.
fc7 = AlexNet(resized, feature_extract=True)

# TODO: Define a new fully connected layer followed by a softmax activation to classify
# the traffic signs. Assign the result of the softmax activation to `probs` below.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imageio.imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imageio.imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

for i in output:
    print(np.argmax(i))
