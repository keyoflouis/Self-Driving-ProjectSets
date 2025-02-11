# import tensorflow as tf
#
# # Create TensorFlow object called tensor
# hello_constant = tf.constant('Hello World!')
#
# with tf.Session() as sess:
#     # Run the tf.constant operation in the session
#     output = sess.run(hello_constant)
#     print(output)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

# TensorFlow 2.x uses eager execution by default, so we can directly print the tensor
print(hello_constant.numpy())
