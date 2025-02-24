import tensorflow as tf

# The file path to load the data
load_file = './model.ckpt'

# 初始w和b
weights = tf.Variable(tf.zeros([2, 3]))
bias = tf.Variable(tf.zeros([3]))

# 从./中加载找到model.ckpt为前缀的文件
checkpoint = tf.train.Checkpoint(weights=weights, bias=bias)
checkpoint.restore(tf.train.latest_checkpoint('./'))

# 打印
print('Weights:')
print(weights.numpy())
print('Bias:')
print(bias.numpy())