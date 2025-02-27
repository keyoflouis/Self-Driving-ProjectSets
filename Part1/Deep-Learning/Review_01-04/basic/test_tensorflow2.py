import tensorflow as tf
import numpy as np

W =np.array([[1.0,2.0],[3.0,4.0]],dtype=np.float32)
b =np.array([0.5,0.5],dtype=np.float32)

x =tf.constant([[1.0,2.0],[3.0,4.0]],dtype=tf.float32)

weight =tf.Variable(W)
bias =tf.Variable(b)

with tf.GradientTape() as tp:
    logits = tf.matmul(x,weight)+bias


gradients =tp.gradient(logits,[weight,bias])
print(gradients[0].numpy())
print(gradients[1].numpy())
