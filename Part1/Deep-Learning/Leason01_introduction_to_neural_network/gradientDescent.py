import numpy as np

# 定义用于激活的Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid函数的导数
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 输入数据
x = np.array([0.1, 0.3])
# 目标值
y = 0.2
# 输入到输出的权重
weights = np.array([-0.8, 0.5])

# 学习率，在权重更新公式中用到的η
learnrate = 0.5

# 神经网络输出（y-hat）
nn_output = sigmoid(np.dot(x, weights))

# 输出误差（y - y-hat）
error = y - nn_output

# 误差项（小写的delta）
error_term = error * sigmoid_prime(np.dot(x, weights))

# 梯度下降步骤
del_w = learnrate * error_term * x
