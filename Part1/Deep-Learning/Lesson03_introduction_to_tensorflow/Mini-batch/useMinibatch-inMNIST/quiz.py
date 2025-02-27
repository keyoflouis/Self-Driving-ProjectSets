import tensorflow as tf
import numpy as np

# 加载数据
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


# 预处理
def preprocess(images, labels):
    features = images.reshape((-1, 784)).astype(np.float32) / 255.0  # 展平并归一化
    labels = tf.keras.utils.to_categorical(labels, 10).astype(np.float32)  # 转one-hot编码
    return features, labels

train_features, train_labels = preprocess(train_images, train_labels)
test_features, test_labels = preprocess(test_images, test_labels)

print(train_labels[0])

# 参数设置
learning_rate = 0.05  # 增大学习率
batch_size = 128
n_input = 784
n_classes = 10
epochs = 5  # 增加训练轮数

# 定义模型参数
weights = tf.Variable(tf.random.normal([n_input, n_classes], dtype=tf.float32))  # 使用 tf.float32
bias = tf.Variable(tf.random.normal([n_classes], dtype=tf.float32))  # 使用 tf.float32

# 优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# 创建数据集
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_features, train_labels)).shuffle(60000).batch(batch_size)

# 训练循环
for epoch in range(epochs):

    for batch_features, batch_labels in train_dataset:
        with tf.GradientTape() as tape:
            # 前向传播
            logits = tf.matmul(batch_features, weights) + bias
            # 计算损失
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=batch_labels, logits=logits
                )
            )
        # 计算梯度
        gradients = tape.gradient(loss, [weights, bias])
        # 更新参数
        optimizer.apply_gradients(zip(gradients, [weights, bias]))

# 测试准确率
test_logits = tf.matmul(test_features, weights) + bias
correct_pred = tf.equal(tf.argmax(test_logits, 1), tf.argmax(test_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('Test Accuracy: {:.3f}'.format(accuracy.numpy()))