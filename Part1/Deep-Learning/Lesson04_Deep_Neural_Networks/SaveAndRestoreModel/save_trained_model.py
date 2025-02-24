import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# 设置超参数
learning_rate = 0.001
n_input = 784  # MNIST 数据输入 (img shape: 28*28)
n_classes = 10  # MNIST 总类别 (0-9 digits)

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape(-1, n_input).astype('float32') / 255
x_test = x_test.reshape(-1, n_input).astype('float32') / 255

# 将标签转换为 one-hot 编码
y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_classes, input_shape=(n_input,), activation='softmax')
])

# 定义损失函数和优化器
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 编译模型
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# 训练模型
batch_size = 128
n_epochs = 100

model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(x_test, y_test))

# 保存模型
save_file = './train_model.ckpt'
model.save_weights(save_file)
print('Trained Model Saved.')