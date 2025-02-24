import tensorflow as tf

# The file path to save the data
save_file = './model.ckpt'

# 使用截断正太分布初始化W维度为2*3和b为1*3，
weights = tf.Variable(tf.random.truncated_normal([2, 3]))
bias = tf.Variable(tf.random.truncated_normal([3]))

# 保存到检查点
checkpoint = tf.train.Checkpoint(weights=weights, bias=bias)
checkpoint.save(save_file)

