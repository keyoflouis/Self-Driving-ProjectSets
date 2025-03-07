## 自动驾驶汽车工程师纳米学位

### 深度学习

#### 项目：构建交通标志识别分类器

在本Notebook中，我们提供了一个分阶段实现功能的模板。如果无法将所有代码包含在Notebook中，请确保通过必要方式导入Python代码。标有"Implementation"的章节是项目实现的主要部分，标有"Optional"的部分为可选实现。

除代码实现外，还需回答与项目相关的问题。每个问题前会有"Question"标识，请仔细阅读并在"Answer:"后的文本框内作答。

---

### 步骤0：加载数据

```python
# 加载pickled数据
import pickle

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
```

---

### 步骤1：数据集概览与探索

数据字典包含四个键值对：
• `features`: 4D数组，存储交通标志图像原始像素（样本数×宽×高×通道）
• `labels`: 1D数组，存储标签ID（signnames.csv提供ID与名称的映射）
• `sizes`: 原始图像尺寸列表（宽, 高）
• `coords`: 图像中标志的边界框坐标列表（原始图像尺寸，已缩放为32×32）

```python
n_train = len(X_train)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(set(y_test))

print("训练样本数 =", n_train)
print("测试样本数 =", n_test)
print("图像尺寸 =", image_shape)
print("类别数 =", n_classes)
```

输出：

```
训练样本数 = 39209
测试样本数 = 12630
图像尺寸 = (32, 32, 3)
类别数 = 43
```

---

### 可视化数据

```python
import random
import matplotlib.pyplot as plt

index = random.randint(0, len(X_train))
plt.imshow(X_train[index].squeeze(), cmap="gray")
print("标签:", y_train[index])
```

![示例图像](输出25号标签图像)

---

### 步骤2：设计与测试模型架构

#### 数据预处理

```python
from sklearn.utils import shuffle

def normalize(image_data):
    """Min-Max归一化到[0.01, 0.99]范围"""
    return 0.01 + (image_data / 255.0) * 0.98

X_train = normalize(X_train)
X_test = normalize(X_test)
X_train, y_train = shuffle(X_train, y_train)
```

---

#### 构建LeNet-5模型

```python
import tensorflow as tf

def LeNet(x):
    # 卷积层1：输入32x32x3 → 输出28x28x6
    conv1 = tf.layers.conv2d(x, 6, 5, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)  # 14x14x6

    # 卷积层2：输出10x10x16
    conv2 = tf.layers.conv2d(pool1, 16, 5, activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # 5x5x16

    # 全连接层
    flatten = tf.contrib.layers.flatten(pool2)
    fc1 = tf.layers.dense(flatten, 120, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 84, activation=tf.nn.relu)
    logits = tf.layers.dense(fc2, 43)
    return logits
```

---

#### 模型训练

```python
EPOCHS = 40
BATCH_SIZE = 120

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)

# 训练过程（略）
```

验证准确率达98%后保存模型：

```
Model saved
```

---

### 关键问题解答

#### 问题1：数据预处理方法

**答**：采用Min-Max归一化将像素值从[0,255]映射到[0.01,0.99]，以标准化数据分布。保留RGB三通道信息以利用颜色特征。

#### 问题2：数据划分策略

**答**：使用`train_test_split`将原始训练集的2%作为验证集。验证集用于监控模型是否过拟合。

#### 问题3：模型架构

**答**：改进版LeNet-5架构：

1. 卷积层（5x5核，ReLU激活）
2. 最大池化（2x2窗口）
3. 第二个卷积层（5x5核，ReLU）
4. 全连接层（120→84个神经元）
5. 输出层（43个类别）

#### 问题4：训练配置

**答**：使用Adam优化器，40个epoch，批量大小120，学习率0.001。未使用dropout，因实验发现会降低性能。

#### 问题5：方案设计思路

**答**：基于LeNet架构进行调优。通过调整超参数和增加训练轮次提升性能，最终验证准确率达98%，测试准确率98%。

---

### 步骤3：新图像测试

#### 测试结果

```python
test_accuracy = evaluate(X_test, y_test)
print("测试准确率 = {:.3f}".format(test_accuracy))  # 输出：0.980
```

#### 新图像预测

加载9张网络图片测试，其中6张预测正确（75%准确率）。错误案例分析：

1. 野生动物标识（数据集未包含）
2. 自行车标识被误判为其他类别

```python
Predicted Labels: [ 8  1 28 33  5 14 18 17  0]
Expected Labels:  ['?', 29, 28, 33, 5, 14, 18, 17, 34]
```

---

### Softmax概率可视化

使用`tf.nn.top_k`展示模型预测置信度：
![top_k可视化](各测试样本前5预测概率分布图)

模型在多数情况下表现出高置信度，但对非常见标识（如野生动物）预测能力有限。未来可通过数据增强和更复杂模型（如ResNet）改进。

---

*注：完成所有实现后，可通过File → Download as → HTML导出完整报告。*

---

## 自动驾驶汽车工程师纳米学位

### 深度学习

#### 项目：构建交通标志识别分类器

本笔记本提供了一个分阶段实现功能的模板，以帮助你顺利完成本项目。如果需要在笔记本外添加额外代码，请确保Python代码已正确导入并包含在提交文件中。标题中标注"Implementation"的部分是项目实现的主要部分，标注"Optional"的部分为可选内容。

除了编写代码，你还需要回答与项目及实现相关的问题。每个需要回答的问题前都会有"Question"标题。请仔细阅读问题，并在以"Answer:"开头的文本框中提供详细回答。项目提交将基于你对问题的回答和提供的代码实现进行评估。

**注意**：使用Shift + Enter快捷键可执行代码和Markdown单元格。双击Markdown单元格可进入编辑模式。

---

### 步骤0：加载数据

```python
# 加载pickle数据
import pickle

# 根据训练集和测试集的保存路径填写此处
training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
```

---

### 步骤1：数据集概览与探索

pickle数据是一个包含4个键值对的字典：  
• `features`：4D数组，包含交通标志图像的原始像素数据（样本数, 宽度, 高度, 通道数）。  
• `labels`：2D数组，包含交通标志的类别ID。`signnames.csv`文件提供了ID到名称的映射。  
• `sizes`：包含元组（宽度, 高度）的列表，表示图像的原始尺寸。  
• `coords`：包含元组（x1, y1, x2, y2）的列表，表示图像中标志的边界框坐标（原始尺寸，pickle数据中的图像已调整为32x32）。  

完成以下基本数据统计：  

```python
# 填写缺失值
n_train = len(X_train)          # 训练样本数
n_test = len(X_test)            # 测试样本数
image_shape = X_train[0].shape  # 图像形状
n_classes = len(set(y_test))    # 类别数量

print("训练样本数 =", n_train)
print("测试样本数 =", n_test)
print("图像数据形状 =", image_shape)
print("类别数 =", n_classes)
```

输出：  

```
训练样本数 = 39209  
测试样本数 = 12630  
图像数据形状 = (32, 32, 3)  
类别数 = 43  
```

#### 可视化数据集

```python
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

index = random.randint(0, len(X_train))
plt.figure(figsize=(1,1))
plt.imshow(X_train[index].squeeze(), cmap="gray")
print("标签:", y_train[index])
```

输出示例：`标签: 25`

---

### 步骤2：设计与测试模型架构

设计并训练一个能识别交通标志的深度学习模型。需考虑以下方面：  
• 神经网络架构  
• 预处理（归一化、RGB转灰度等）  
• 各类别样本数量不均衡问题  
• 数据增强  

#### 数据预处理

```python
from sklearn.utils import shuffle

# 图像数据归一化
def normalize(image_data):
    a, b = 0.01, 0.99
    return a + (image_data / 255.0) * (b - a)

X_train = normalize(X_train)
X_test = normalize(X_test)
X_train, y_train = shuffle(X_train, y_train)

# 划分验证集
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=0.02, random_state=42)
```

#### 实现LeNet-5架构

```python
from tensorflow.contrib.layers import flatten
import tensorflow as tf

model_name = 'lenet_report'
EPOCHS = 40
BATCH_SIZE = 120

def LeNet(x):    
    # 超参数
    mu, sigma = 0, 0.01

    # 卷积层1：输入32x32x3，输出28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,3,6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1,1,1,1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # 卷积层2：输出10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1,1,1,1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # 全连接层
    fc0 = flatten(conv2)
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    # 输出层
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84,43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
```

#### 训练与评估

```python
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_op = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()

# 训练过程
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, len(X_train), BATCH_SIZE):
            batch_x = X_train[offset:offset+BATCH_SIZE]
            batch_y = y_train[offset:offset+BATCH_SIZE]
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = sess.run(accuracy_op, feed_dict={x: X_validation, y: y_validation})
        print(f"EPOCH {i+1} ... 验证准确率: {validation_accuracy:.3f}")

    saver.save(sess, './models/'+model_name)
    print("模型已保存")
```

---

### 步骤3：在新图像上测试模型

从网络或现实场景中拍摄至少五张交通标志图像，使用模型进行分类。  

#### 测试结果示例

```python
# 加载测试图像
from skimage import io
test_images = [...]  # 图像路径列表
test_imgs = np.array([io.imread(img) for img in test_images])
test_imgs = normalize(test_imgs.reshape((-1,32,32,3)))

# 预测
with tf.Session() as sess:
    saver.restore(sess, './models/lenet_report')
    predictions = sess.run(tf.argmax(logits, 1), feed_dict={x: test_imgs})
    print("预测结果:", predictions)
    print("真实标签:", [29, 28, 33, 5, 14, 18, 17, 34])
```

---

### 问题与回答

**问题1**：数据预处理步骤及原因  
**答**：使用Min-Max归一化将像素值缩放到[0.01, 0.99]，以加速训练并提高模型稳定性。保留RGB三通道以利用颜色信息。

**问题2**：数据划分策略  
**答**：使用`train_test_split`从训练集中划分2%作为验证集，用于监控训练过程中的过拟合。

**问题3**：最终模型架构  
**答**：基于LeNet-5，包含两个卷积层、两个池化层、三个全连接层，输出43维logits。

**问题4**：训练参数  
**答**：使用Adam优化器，学习率0.001，批量大小120，训练40轮次，最终验证准确率98%。

**问题5**：改进思路  
**答**：尝试数据增强（旋转、平移）、调整网络深度、使用更复杂的架构（如ResNet）或自动化超参数优化。

**问题6-8**：新图像测试结果分析  
**答**：模型对部分未见过的标志（如野生动物穿越）分类错误，但在已知类别上表现良好。通过softmax概率可视化，模型对多数预测结果置信度较高。

---

交通标志识别分类器项目报告

# 自动驾驶汽车工程师纳米学位

## 深度学习

### 项目：构建交通标志识别分类器

本文档提供了一个模板，供您分阶段实现所需的功能，以成功完成此项目。如果需要添加无法包含在笔记本中的代码，请确保Python代码成功导入并包含在您的提交中。标题以“实现”开头的章节表示您应开始为项目实现代码的位置。请注意，某些实现章节是可选的，并将在标题中标记为“可选”。

除了实现代码，还将提出与项目和实现相关的回答问题。每个回答问题的章节前面都有一个“问题”标题。仔细阅读每个问题，并在以“回答：”开头的文本框中提供详尽的回答。您的项目提交将根据您对每个问题的回答和提供的实现来评估。

**注意**：可以使用Shift + Enter快捷键来执行代码和Markdown单元格。此外，Markdown单元格通常可以通过双击单元格进入编辑模式进行编辑。

## 第0步：加载数据

```python
# 加载泡菜数据
import pickle

# TODO：根据保存训练和测试数据的位置填写此内容
训练文件 = 'train.p'
测试文件 = 'test.p'

with open(训练文件, mode='rb') as f:
    训练数据 = pickle.load(f)
with open(测试文件, mode='rb') as f:
    测试数据 = pickle.load(f)

X_train, y_train = 训练数据['features'], 训练数据['labels']
X_test, y_test = 测试数据['features'], 测试数据['labels']
```

## 第1步：数据集摘要和探索

泡菜数据是一个包含4个键值对的字典：

- `'features'` 是一个4D数组，包含交通标志图像的原始像素数据，形状为（示例数，宽度，高度，通道数）。
- `'labels'` 是一个2D数组，包含交通标志的标签/类ID。文件signnames.csv包含每个ID对应的名称映射。
- `'sizes'` 是一个包含元组的列表，（宽度，高度）表示图像的原始宽度和高度。
- `'coords'` 是一个包含元组的列表，（x1, y1, x2, y2）表示图像中标志周围的边界框的坐标。这些坐标基于原始图像。泡菜数据包含这些图像的调整大小版本（32x32）。

完成以下基本数据摘要。

### 将每个问号替换为适当的值。

```python
# TODO：训练样本数量
n_train = len(X_train)

# TODO：测试样本数量
n_test = len(X_train)

# 交通标志图像的形状
image_shape = X_train[0].shape

# 数据集中唯一的类别/标签数量
n_classes = len(set(y_test))

print("训练样本数量 =", n_train)
print("测试样本数量 =", n_test)
print("图像数据形状 =", image_shape)
print("类别数量 =", n_classes)
```

**输出结果**：

```
训练样本数量 = 39209
测试样本数量 = 12630
图像数据形状 = (32, 32, 3)
类别数量 = 43
```

### 设置验证特征

使用泡菜文件可视化德国交通标志数据集。可以使用Matplotlib示例和库页面作为在Python中进行可视化的资源。

**注意**：建议从简单的内容开始。如果希望做更多，完成其他章节后再回来做。

### 数据探索可视化如下。

```python
import random
import numpy as np
import matplotlib.pyplot as plt

# 可视化将在笔记本中显示。
%matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])
```

**输出结果**：

```
25
```

## 第2步：设计和测试模型架构

设计和实现一个深度学习模型，以学习识别交通标志。在德国交通标志数据集上训练和测试模型。

考虑此问题时需注意的各个要素：

- 神经网络架构
- 尝试预处理技术（归一化、彩色转灰度等）
- 每个标签的样本数量（有些标签的样本比其他标签多）
- 生成假数据

这里有一个在此问题上的已发布基线模型示例，虽然无需熟悉论文中使用的方法，但尝试阅读此类论文是个好习惯。

**注意**：在CNN课程最后展示的LeNet-5实现是一个很好的起点。你需要更改类别数量和可能的预处理，但除此之外是可以直接使用的！

### 预处理数据

打乱训练数据。

```python
from sklearn.utils import shuffle

# 实现图像数据的最小-最大缩放
def 归一化(image_data):
    a = 0.01
    b = 0.99
    color_min = 0.0
    color_max = 255.0
    return a + ( ( (image_data - color_min) * (b - a) ) / (color_max - color_min))

# 归一化训练和测试特征
X_train = 归一化(X_train)
X_test = 归一化(X_test)

X_train, y_train = shuffle(X_train, y_train)
```

```python
from sklearn.model_selection import train_test_split

X_train = np.append(X_train, X_test, axis=0)
y_train = np.append(y_train, y_test, axis=0)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.02,
    random_state=42)
```

## 使用TensorFlow配置

EPOCH和BATCH_SIZE值会影响训练速度和模型准确性。

实现LeNet-5

实现LeNet-5神经网络架构。

输入

LeNet架构接受一个32x32xC的图像作为输入，其中C是颜色通道数。由于图像是彩色的，因此在这种情况下C是3。

架构

第1层：卷积层。输出形状应为28x28x6。

激活函数。你可以选择激活函数。

池化。输出形状应为14x14x6。

第2层：卷积层。输出形状应为10x10x16。

激活函数。你可以选择激活函数。

池化。输出形状应为5x5x16。

展平。将最终池化层的输出形状展平，使其为一维而不是三维。最容易的方法是使用已经为你导入的tf.contrib.layers.flatten。

第3层：全连接层。这应该有120个输出。

激活函数。你可以选择激活函数。

第4层：全连接层。这应该有84个输出。

激活函数。你可以选择激活函数。

第5层：全连接层（逻辑单元）。这应该有43个输出。

输出

返回第二层全连接层的结果。

```python
from tensorflow.contrib.layers import flatten
import tensorflow as tf

模型名称 = 'lenet_report'

EPOCHS = 40
BATCH_SIZE = 120

def LeNet(x):
    # 超参数
    mu = 0
    sigma = 0.01
    keep_prob = 1

    # 第一层：卷积层。输入=32x32x3。输出=28x28x6。
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # 解决方案：激活函数。
    conv1 = tf.nn.relu(conv1)

    # 解决方案：池化。输入=28x28x6。输出=14x14x6。
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 解决方案：第二层：卷积层。输出=10x10x16。
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # 解决方案：激活函数。
    conv2 = tf.nn.relu(conv2)

    # 解决方案：池化。输入=10x10x16。输出=5x5x16。
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 解决方案：展平。输入=5x5x16。输出=400。
    fc0 = flatten(conv2)

    # 解决方案：第三层：全连接层。输入=400。输出=120。
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # 解决方案：激活函数。
    fc1 = tf.nn.relu(fc1)

    # 解决方案：第四层：全连接层。输入=120。输出=84。
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # 解决方案：激活函数。
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # 第五层：全连接层。输入=84。输出=43。
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
```

特征和标签
训练LeNet对输入数据进行分类。

`x` 是一个占位符，用于一批输入图像。`y` 是一个占位符，用于一批输出标签。

```python
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
```

### 在此处训练模型。

```python
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("开始训练...")
    print()

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("验证集准确率 = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './models/'+模型名称)
    print("模型已保存")
```

训练输出结果：

```
开始训练...

EPOCH 1 ...
验证集准确率 = 0.286

EPOCH 2 ...
验证集准确率 = 0.431

EPOCH 3 ...
验证集准确率 = 0.481

EPOCH 4 ...
验证集准确率 = 0.600

EPOCH 5 ...
验证集准确率 = 0.740

EPOCH 6 ...
验证集准确率 = 0.805

EPOCH 7 ...
验证集准确率 = 0.868

EPOCH 8 ...
验证集准确率 = 0.881

EPOCH 9 ...
验证集准确率 = 0.928

EPOCH 10 ...
验证集准确率 = 0.937

EPOCH 11 ...
验证集准确率 = 0.940

EPOCH 12 ...
验证集准确率 = 0.946

EPOCH 13 ...
验证集准确率 = 0.949

EPOCH 14 ...
验证集准确率 = 0.955

EPOCH 15 ...
验证集准确率 = 0.952

EPOCH 16 ...
验证集准确率 = 0.952

EPOCH 17 ...
验证集准确率 = 0.959

EPOCH 18 ...
验证集准确率 = 0.960

EPOCH 19 ...
验证集准确率 = 0.969

EPOCH 20 ...
验证集准确率 = 0.960

EPOCH 21 ...
验证集准确率 = 0.962

EPOCH 22 ...
验证集准确率 = 0.969

EPOCH 23 ...
验证集准确率 = 0.973

EPOCH 24 ...
验证集准确率 = 0.968

EPOCH 25 ...
验证集准确率 = 0.971

EPOCH 26 ...
验证集准确率 = 0.974

EPOCH 27 ...
验证集准确率 = 0.977

EPOCH 28 ...
验证集准确率 = 0.983

EPOCH 29 ...
验证集准确率 = 0.976

EPOCH 30 ...
验证集准确率 = 0.970

EPOCH 31 ...
验证集准确率 = 0.971

EPOCH 32 ...
验证集准确率 = 0.971

EPOCH 33 ...
验证集准确率 = 0.980

EPOCH 34 ...
验证集准确率 = 0.978

EPOCH 35 ...
验证集准确率 = 0.980

EPOCH 36 ...
验证集准确率 = 0.981

EPOCH 37 ...
验证集准确率 = 0.974

EPOCH 38 ...
验证集准确率 = 0.971

EPOCH 39 ...
验证集准确率 = 0.978

EPOCH 40 ...
验证集准确率 = 0.971

模型已保存
```

问题1：
描述你是如何预处理数据的。你为什么选择这种技术？

回答：
对于预处理，我使用了最小-最大归一化。我将测试和训练数据进行了归一化处理。我将图像数据的0-255值归一化到0-1，这样做可以让我们接近0均值和相等方差。此外，我没有将图像转换为灰度图，以保留3个通道的所有特性，这样在进行卷积时可以提取更多信息。

问题2：
描述你是如何设置模型的训练、验证和测试数据的。可选：如果你生成了额外的数据，你是如何生成数据的？你为什么生成数据？新数据集（包含生成的数据）与原始数据集有什么不同？

回答：
我使用了测试和训练拆分函数将训练数据拆分为训练和验证数据。我使用了20%的训练数据进行验证。这很有必要，因为我们需要在训练后验证训练结果，以测量预测准确率。

问题3：
你的最终架构是什么样的？（模型类型、层、大小、连接等）。有关如何使用TensorFlow构建深度神经网络的参考，请参阅课堂上的“深度神经网络在TensorFlow中”。

回答：
我重用了LeNet实验，也就是LeNet-5神经网络架构。五层结构如下：

- 第一层：卷积层。输出形状为28x28x6。
- 激活函数：我使用了RELU激活层。
- 池化：我使用了最大池化，输出形状为14x14x6。
- 第二层：卷积层。输出形状为10x10x16。
- 激活函数：我使用了RELU激活层。
- 池化：我使用了最大池化，输出形状为5x5x16。
- 展平：将最终池化层的输出形状展平，使其为一维而不是三维。我使用了tf.contrib.layers.flatten。
- 第三层：是一个具有120个输出的全连接层。
- 激活函数：我使用了RELU激活层。
- 第四层：是一个具有84个输出的全连接层。
- 激活函数：我使用了RELU激活层。
- 第五层：具有43个输出的全连接层（逻辑单元）。

问题4：
你是如何训练模型的？（优化器类型、批量大小、轮数、超参数等）

回答：
EPOCH和BATCH_SIZE值会影响训练速度和模型准确性。我尝试了各种轮数、批量大小和学习率的组合。最终，我在40轮、批量大小为150和学习率为0.01的情况下获得了98%的准确率。我没有修改其他超参数。尽管我尝试了0.5的dropout，但并没有得到好的结果。

问题5：
你是如何解决这个问题的？这可能是一个反复试验的过程，在这种情况下，概述你采取的步骤以达到最终解决方案，以及你选择这些步骤的原因。或许你的解决方案涉及一个已经很成熟的实现或架构。在这种情况下，讨论为什么你认为这对当前问题合适。

回答：
我遵循了卷积神经网络中讨论的LeNet架构方法。我选择这种方法是因为它看起来是一种更有效的训练方法。我花了很多时间尝试提高训练准确率。一些预处理操作，如归一化输入，然后我使用了涉及卷积网络、RELU和最大池化的LeNet。我花了很多时间调整超参数，如dropout和标准差等。我还尝试了不同的轮数、学习率和批量大小的组合。在获得98%的训练准确率后，我认为我的方法足够好，可以用于这个项目。

```python
with tf.Session() as sess:
    print('正在加载'+模型名称+'...')
    saver.restore(sess, './models/'+模型名称)
    print('已加载')
    test_accuracy = evaluate(X_test, y_test)
    print("测试准确率 = {:.3f}".format(test_accuracy))
```

输出结果：

```
正在加载 lenet_report...
已加载
测试准确率 = 0.980
```

## 第三步：在新图像上测试模型

从网络上或你周围拍摄几张交通标志的照片（至少5张），在你的计算机上运行它们通过分类器，以产生示例结果。分类器可能无法识别一些地方标志，但可能证明这是有趣的。

你可能会发现 signnames.csv 很有用，因为它包含从类ID（整数）到实际标志名称的映射。

实现
使用代码单元格（如果需要，可以使用多个代码单元格）来实现项目的第一个步骤。一旦你完成了实现并对结果满意，请确保在报告后续提出的问题时给出详细的答案。

问题6：
选择五张交通标志的候选图像，并在报告中提供它们。是否有任何特定的图像质量可能会使分类变得困难？绘制图像可能会有所帮助。

回答：
我使用了9张图像（感谢Tyler Lanigan提供的图像）。

测试标志及其类别的摘要如下表所示：

测试图像    标志类别
1    野生动物过道 - ？
2    自行车过道 - 29
3    儿童过道 - 28
4    前方右转 - 33
5    限速（80公里/小时） - 5
6    停车 - 14
7    一般谨慎 - 18
8    禁止进入 - 17
9    前方左转 - 34
在这些图像中，第一张图像不属于任何标志类别，而且有些图像不在数据集中。因此，这使得模型更难分类。但我期待接近的猜测！

```python
# 加载测试图像
from skimage import io
import numpy as np
import os

images = os.listdir("testImages/")
images.sort()
num_imgs = len(images)
test_imgs = np.uint8(np.zeros((num_imgs, 32, 32, 3)))
labels = ['?', 29, 28, 33, 5, 14, 18, 17, 34]

for i, j in enumerate(images):
    image = io.imread('./testImages/' + j)
    test_imgs[i] = image

# 归一化训练和测试特征
test_imgs = 归一化(test_imgs.reshape((-1, 32, 32, 3)).astype(np.float32))
```

输出结果：

```
['?', 29, 28, 33, 5, 14, 18, 17, 34]
```

```python
import matplotlib.pyplot as plt
f, ax = plt.subplots(num_imgs, 1)
for i in range(num_imgs):
    ax[i].imshow(test_imgs[i])
    plt.setp(ax[i].get_xticklabels(), visible=False)
    plt.setp(ax[i].get_yticklabels(), visible=False)

plt.show()
```

问题7：
与在数据集上测试相比，你的模型在拍摄的图片上表现如何？最简单的方法是检查预测的准确性。例如，如果模型正确预测了5个标志中的1个，那么它是20%准确。

请注意：你可以通过使用signnames.csv（同一目录）来手动检查准确率。这个文件将类ID（0-42）映射到相应的标志名称。因此，你可以取模型输出的类ID，在signnames.csv中查找名称，看看它是否与图像中的标志匹配。

回答：
如以下结果所示，我们可以看到图像3、4、5、6、7、8被正确预测。第一个图像的标签不在提供的csv文件中。2和9没有被正确预测。因此我们的模型有75%的准确率[(6/8)*100]。这正如预期，因为有些图像不在我们的数据集中。然而，我的准确率远低于测试准确率0.98。可以通过在LeNet中添加更多层或使用更好的架构如Keras，或者简单地通过移除显式学习率并让Adam Optimizer选择一个来进一步改进模型！

```python
import tensorflow as tf

模型名称 = 'lenet_report'

predictions = tf.nn.softmax(logits)

def classify_images(X_data):
    sess = tf.get_default_session()
    pred_vals = sess.run(predictions, feed_dict={x: X_data})
    return pred_vals

with tf.Session() as sess:
    print('正在加载'+模型名称+'...')
    saver.restore(sess, './models/'+模型名称)
    predictions = classify_images(test_imgs)
    top_k = sess.run(tf.nn.top_k(predictions, 5, sorted=True))
    print("预测标签：", np.argmax(predictions, 1))
    print("预期标签： ", labels)
```

输出结果：

```
正在加载 lenet_report...
预测标签： [ 8  1 28 33  5 14 18 17  0]
预期标签：  ['?', 29, 28, 33, 5, 14, 18, 17, 34]
```

问题8：
使用模型的softmax概率来可视化其预测的确定性，`tf.nn.top_k`在这里可能会有帮助。哪些预测是模型确定的？不确定的？如果模型的初始预测是错误的，正确的预测是否出现在前k个预测中？（k最多应该是5）

`tf.nn.top_k`将返回前k个预测的值和索引（类ID）。所以如果k=3，对于每个标志，它将返回3个最大概率（43个中）和对应的类ID。

例如，取这个numpy数组：

```python
# (5, 6)数组
a = np.array([[0.24879643, 0.07032244, 0.12641572, 0.34763842, 0.07893497, 0.12789202],
       [0.28086119, 0.27569815, 0.08594638, 0.0178669 , 0.18063401, 0.15899337],
       [0.26076848, 0.23664738, 0.08020603, 0.07001922, 0.1134371 , 0.23892179],
       [0.11943333, 0.29198961, 0.02605103, 0.26234032, 0.1351348 , 0.16505091],
       [0.09561176, 0.34396535, 0.0643941 , 0.16240774, 0.24206137, 0.09155967]])

运行它通过`sess.run(tf.nn.top_k(tf.constant(a), k=3))`的结果是：

TopKV2(values = array([[0.34763842, 0.24879643, 0.12789202],
[0.28086119, 0.27569815, 0.18063401],
[0.26076848, 0.23892179, 0.23664738],
[0.29198961, 0.26234032, 0.16505091],
[0.34396535, 0.24206137, 0.16240774]]), indices = array([[3, 0, 5],
[0, 1, 4],
[0, 5, 1],
[1, 3, 5],
[1, 4, 3]], dtype=int32))

查看第一行我们得到[0.34763842, 0.24879643, 0.12789202]，你可以确认这些是a中的三个最大概率。你还会注意到[3, 0, 5]是对应的索引。

回答：
在五个可视化的预测中，模型错误预测了第一和第二个标签。第一张预测图像是一个马穿过道路的图片。由于在德国交通标志数据集中没有"马过道"标志，模型难以识别是可以预料的，然而，我会将正确的预测视为野生动物过道，或31号类别。
根据上述可视化，在所有五种情况下，神经网络模型以最高概率（几乎100%）做出第一选择，其他四个选择几乎可以忽略不计。对我来说，这看起来有点警示。因此，需要进一步调查以理解这种行为。


```python
N = 5

ind = np.arange(N)  # 值的位置

for i in range(5):
    plt.figure(i)
    values = top_k[0][i]
    plt.bar(range(N), values, 0.40, color='g')
    plt.ylabel('概率')
    plt.xlabel('类标签')
    plt.title('测试图像{}的前{}个Softmax概率'.format(str(i+1), N))
    plt.xticks(ind+0.40, tuple(top_k[1][i]))

plt.show()
```

**注**：一旦你完成了所有代码实现，并成功回答了上述问题，你就可以通过导出iPython Notebook为HTML文档来最终完成你的工作。你可以通过菜单栏中的"文件 -> 下载为 -> HTML (.html)"来完成此操作。请将完成的文档连同此笔记本一起提交。
