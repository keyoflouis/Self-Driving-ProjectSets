"""
设置步长、填充和滤波器权重/偏置，使得输出形状为 (1, 2, 2, 3)。
"""
import tensorflow as tf
import numpy as np

# 输入形状需为 4D (batch_size, height, width, depth) = (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)


def conv2d(input):
    # TODO 定义滤波器权重（形状为 [height, width, input_depth, output_depth]）
    F_W = tf.Variable(tf.random.truncated_normal([2,2,1,3]))
    # TODO 定义滤波器偏置（每个输出通道一个偏置）
    F_b = tf.Variable(tf.zeros([3]))

    # 设置步长（[batch, height, width, depth]）
    strides =[1,2,2,1]

    # 设置填充模式为 VALID（无填充）
    padding ='VALID'

    # 执行卷积运算并添加偏置
    return tf.nn.conv2d(input, F_W, strides, padding) + F_b

out = conv2d(X)

# 验证输出形状
print(out.shape)
# 输出应为 (1, 2, 2, 3)