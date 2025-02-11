# # Solution is available in the other "sandbox_solution.py" tab
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# from quiz import get_weights, get_biases, linear
#
#
# def mnist_features_labels(n_labels):
#     """
#     Gets the first <n> labels from the MNIST dataset
#     :param n_labels: Number of labels to use
#     :return: Tuple of feature list and label list
#     """
#     mnist_features = []
#     mnist_labels = []
#
#     mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)
#
#     # In order to make quizzes run faster, we're only looking at 10000 images
#     for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):
#
#         # Add features and labels if it's for the first <n>th labels
#         if mnist_label[:n_labels].any():
#             mnist_features.append(mnist_feature)
#             mnist_labels.append(mnist_label[:n_labels])
#
#     return mnist_features, mnist_labels
#
#
# # Number of features (28*28 image is 784 features)
# n_features = 784
# # Number of labels
# n_labels = 3
#
# # Features and Labels
# features = tf.placeholder(tf.float32)
# labels = tf.placeholder(tf.float32)
#
# # Weights and Biases
# w = get_weights(n_features, n_labels)
# b = get_biases(n_labels)
#
# # Linear Function xW + b
# logits = linear(features, w, b)
#
# # Training data
# train_features, train_labels = mnist_features_labels(n_labels)
#
# with tf.Session() as session:
#     # TODO: Initialize session variables
#     session.run(tf.global_variables_initializer())
#     # Softmax
#     prediction = tf.nn.softmax(logits)
#
#     # Cross entropy
#     # This quantifies how far off the predictions were.
#     # You'll learn more about this in future lessons.
#     cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
#
#     # Training loss
#     # You'll learn more about this in future lessons.
#     loss = tf.reduce_mean(cross_entropy)
#
#     # Rate at which the weights are changed
#     # You'll learn more about this in future lessons.
#     learning_rate = 0.08
#
#     # Gradient Descent
#     # This is the method used to train the model
#     # You'll learn more about this in future lessons.
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#
#     # Run optimizer and get loss
#     _, l = session.run(
#         [optimizer, loss],
#         feed_dict={features: train_features, labels: train_labels})
#
# # Print loss
# print('Loss: {}'.format(l))

import tensorflow as tf
import numpy as np


def get_weights(n_features, n_labels):
    return tf.Variable(tf.random.normal([n_features, n_labels], stddev=0.1))


def get_biases(n_labels):
    return tf.Variable(tf.zeros([n_labels]))


def linear(x, w, b):
    return tf.matmul(x, w) + b


def mnist_features_labels(n_labels):
    # 加载MNIST数据集
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # 归一化并reshape
    train_images = train_images.reshape((-1, 784)).astype('float32') / 255.0

    # 转换为one-hot编码
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)

    # 打乱数据并取前10000个样本
    indices = np.arange(len(train_images))
    np.random.shuffle(indices)
    sampled_indices = indices[:10000]

    sampled_images = train_images[sampled_indices]
    sampled_labels = train_labels[sampled_indices]

    # 筛选标签属于前n_labels类的样本
    mask = np.any(sampled_labels[:, :n_labels], axis=1)
    filtered_images = sampled_images[mask]
    filtered_labels = sampled_labels[mask][:, :n_labels]

    return filtered_images, filtered_labels


# 设置参数
n_features = 784
n_labels = 3

# 获取数据
train_features, train_labels = mnist_features_labels(n_labels)

# 转换为TensorFlow张量
features = tf.convert_to_tensor(train_features, dtype=tf.float32)
labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)

# 初始化模型参数
w = get_weights(n_features, n_labels)
b = get_biases(n_labels)

# 设置优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.08)

# 训练步骤
with tf.GradientTape() as tape:
    logits = linear(features, w, b)
    # 计算损失（使用内置交叉熵函数）
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# 计算梯度并更新参数
gradients = tape.gradient(loss, [w, b])
optimizer.apply_gradients(zip(gradients, [w, b]))

# 打印损失
print('Loss: {:.4f}'.format(loss.numpy()))