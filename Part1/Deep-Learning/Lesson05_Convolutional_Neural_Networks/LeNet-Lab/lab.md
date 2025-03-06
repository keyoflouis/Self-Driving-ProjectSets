# LeNet 实验室解决方案

## 加载数据

加载 TensorFlow 预加载的 MNIST 数据集。 
你不需要修改这一部分。

```python
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("图像形状: {}".format(X_train[0].shape))
print()
print("训练集:   {} 样本".format(len(X_train)))
print("验证集: {} 样本".format(len(X_validation)))
print("测试集:       {} 样本".format(len(X_test)))
```

TensorFlow 预加载的 MNIST 数据集是 28x28x1 的图像。

然而，LeNet 架构只接受 32x32xC 的图像，其中 C 是颜色通道的数量。 
为了将 MNIST 数据集重新格式化为 LeNet 可接受的形状，我们在图像的顶部和底部各填充两行零，在左侧和右侧各填充两列零（28+2+2 = 32）。 
你不需要修改这一部分。

```python
import numpy as np

# 用零填充图像
X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print("更新后的图像形状: {}".format(X_train[0].shape))
```

## 可视化数据

查看数据集中的一个样本。 
你不需要修改这一部分。

```python
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])
```

## 数据预处理

打乱训练数据。 
你不需要修改这一部分。

```python
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
```

## 设置 TensorFlow

`EPOCH` 和 `BATCH_SIZE` 的值会影响训练速度和模型精度。 
你不需要修改这一部分。

```python
import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128
```

## 实现 LeNet-5

实现 [LeNet-5](http://yann.lecun.com/exdb/lenet/) 神经网络架构。 
这是唯一需要你编辑的部分。

### 输入

LeNet 架构接受 32x32xC 的图像作为输入，其中 C 是颜色通道的数量。由于 MNIST 图像是灰度图像，因此 C 在这里为 1。

### 架构

**第 1 层：卷积层。** 输出形状应为 28x28x6。 
**激活函数。** 你可以选择激活函数。 
**池化。** 输出形状应为 14x14x6。 
**第 2 层：卷积层。** 输出形状应为 10x10x16。 
**激活函数。** 你可以选择激活函数。 
**池化。** 输出形状应为 5x5x16。 
**展平。** 将最终池化层的输出形状展平为 1D，而不是 3D。最简单的方法是使用 `tf.contrib.layers.flatten`，它已经为你导入。 
**第 3 层：全连接层。** 应有 120 个输出。 
**激活函数。** 你可以选择激活函数。 
**第 4 层：全连接层。** 应有 84 个输出。 
**激活函数。** 你可以选择激活函数。 
**第 5 层：全连接层（输出）。** 应有 10 个输出。

### 输出

返回第 2 个全连接层的结果。

```python
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # 超参数
    mu = 0
    sigma = 0.1

    # 第 1 层：卷积层。输入 = 32x32x1。输出 = 28x28x6。
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # 激活函数
    conv1 = tf.nn.relu(conv1)

    # 池化。输入 = 28x28x6。输出 = 14x14x6。
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 第 2 层：卷积层。输出 = 10x10x16。
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # 激活函数
    conv2 = tf.nn.relu(conv2)

    # 池化。输入 = 10x10x16。输出 = 5x5x16。
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 展平。输入 = 5x5x16。输出 = 400。
    fc0 = flatten(conv2)

    # 第 3 层：全连接层。输入 = 400。输出 = 120。
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # 激活函数
    fc1 = tf.nn.relu(fc1)

    # 第 4 层：全连接层。输入 = 120。输出 = 84。
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # 激活函数
    fc2 = tf.nn.relu(fc2)

    # 第 5 层：全连接层。输入 = 84。输出 = 10。
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
```

## 特征和标签

训练 LeNet 对 [MNIST](http://yann.lecun.com/exdb/mnist/) 数据进行分类。 
`x` 是输入图像的占位符。 
`y` 是输出标签的占位符。 
你不需要修改这一部分。

```python
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
```

## 训练流程

创建一个训练流程，使用模型对 MNIST 数据进行分类。 
你不需要修改这一部分。

```python
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

## 模型评估

评估模型在给定数据集上的损失和精度。 
你不需要修改这一部分。

```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

## 训练模型

将训练数据通过训练流程以训练模型。 
在每个 epoch 之前，打乱训练集。 
在每个 epoch 之后，测量验证集的损失和精度。 
训练完成后保存模型。 
你不需要修改这一部分。

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("正在训练...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("验证精度 = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print("模型已保存")
```

## 评估模型

当你对模型完全满意后，使用测试集评估模型的性能。 
请确保只做一次！ 
如果你在测试集上测量训练模型的性能，然后改进模型，再次测量测试集上的性能，这将使测试结果无效。你无法真正衡量模型在真实数据上的表现。 
你不需要修改这一部分。

```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("测试精度 = {:.3f}".format(test_accuracy))
```

### tensorflow 2.x风格

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# 加载数据
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 预处理
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255

# 划分验证集
validation_split = 5000
X_train, y_train = train_images[:-validation_split], train_labels[:-validation_split]
X_validation, y_validation = train_images[-validation_split:], train_labels[-validation_split:]
X_test, y_test = test_images, test_labels

# 填充到32x32
X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# 断言检查
assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print("\n图像形状: {}".format(X_train[0].shape))
print("训练集:   {} 样本".format(len(X_train)))
print("验证集: {} 样本".format(len(X_validation)))
print("测试集:       {} 样本".format(len(X_test)))

# 可视化数据
index = random.randint(0, len(X_train)-1)
image = X_train[index].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print("样本标签:", y_train[index])

# 构建LeNet模型
def LeNet():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, (5,5), activation='relu', input_shape=(32,32,1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Conv2D(16, (5,5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

model = LeNet()

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练参数
EPOCHS = 10
BATCH_SIZE = 128

# 训练模型
history = model.fit(X_train, y_train, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE,
                    validation_data=(X_validation, y_validation),
                    verbose=1)

# 保存模型    
model.save('lenet_model')

# 评估测试集
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\n测试精度 = {:.3f}".format(test_acc))
```
