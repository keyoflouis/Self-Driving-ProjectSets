import tensorflow as tf

# The file path to load the data
load_file = './model.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.zeros([2, 3]))
bias = tf.Variable(tf.zeros([3]))

# Load the model
checkpoint = tf.train.Checkpoint(weights=weights, bias=bias)
checkpoint.restore(tf.train.latest_checkpoint('./'))

# Show the values of weights and bias
print('Weights:')
print(weights.numpy())
print('Bias:')
print(bias.numpy())