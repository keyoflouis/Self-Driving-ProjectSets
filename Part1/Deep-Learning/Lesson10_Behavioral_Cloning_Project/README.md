### 环境

drive.py 

socketio库编写的服务器与udacity模拟器的socketio版本不同时容易出现不兼容。

- 我的drive.py实现中使用到了`tensorflow 2.x` 的API

- [socketio环境参考](https://zhuanlan.zhihu.com/p/356440288)  

---

# 最终结果：use_mut_train_o3.py

- [ ] 未完

- [x] 需要熟悉数据集管道的构建

---

tarin.py 自定义神经网络

resize_train.py 自定义神经网络+裁剪数据

use_mutcamera_train.py 使用了多个摄像头,

use_mut_train_o1.py 使用了管道预处理数据集,NVIDIA的神经网络

我注意到drive.py对图片进行过预处理，但我的神经网络内部已经处理过图片，修改后重新尝试

use_mut_train_o2.py 使用特征融合,采用多次训练，在最后一个弯道表现不佳。

use_mut_train_o2_5.py deepseek添加了注意力机制,

use_mut_train_o3.py 成功在0.18的油门下行驶一圈

use_mut_train_o3_5.py （失败）

use_mut_train_o4.py 尝试对数据分布进行处理（失败）

----

train2.py 去掉了速度特征，尝试更均衡的数据分布：直行 30 % ，大转弯9%。

模型训练过程损失下降顺利，但实际运行失败

train3.py 尝试调整模型配置，使用余弦退火，运行失败

train4.py 调整数据集分布，失败

----

2025.3.22

行为克隆项目卡着了

问题1，训练数据目前已经基本解决，训练数据仍可以进一步加强，采用各种光照。

问题2，模型采用NVIDIA的架构合理么（大概是的）。那么我对模型不够熟悉，无法更改模型结构

（当前关键问题）问题3，模型编译参数，优化器配置等，我不熟悉（当前关键问题），试训练效率低下

问题4，模型的单次预测我无法知晓具体内部内容

----

2025.3.23

解决问题2，问题3，问题4。

若是仍然无法做到，就不继续纠结其任何他的问题，去看别人的答案

- [x] 尝试分箱取样

# [原文](https://navoshta.com/end-to-end-deep-learning/)

## 自动驾驶汽车的端到端学习

我假设你已经对神经网络和正则化有一定了解，因此不会深入探讨它们的背景和工作原理。我使用Keras作为机器学习框架，后端为TensorFlow，并依赖一些库如`numpy`、`pandas`和`scikit-image`。如果你想跟随教程操作，可能需要一台具有CUDA兼容GPU的机器。

训练模型在模拟器中驾驶汽车是Udacity自动驾驶汽车纳米学位课程中的一个任务，但即使没有该背景知识，这里描述的概念也容易理解。

## 数据集

提供的驾驶模拟器有两个不同的赛道。其中一个用于收集训练数据，另一个从未被模型见过，作为测试集的替代。

### 数据收集

驾驶模拟器会保存来自三个前置“摄像头”的帧，记录汽车视角的数据，以及各种驾驶统计数据，如油门、速度和转向角。我们将使用摄像头数据作为模型输入，并期望它预测范围在`[-1, 1]`内的转向角。

我收集了一个数据集，包含大约1小时在一个给定赛道上的驾驶数据。这包括两种模式的驾驶数据：“平滑”模式（在整个赛道上保持在道路中间）和“恢复”模式（让汽车偏离中心，然后进行干预以将其引导回中间）。

### 平衡数据集

不出所料，结果数据集非常不平衡，包含大量转向角接近`0`的样本（例如，当方向盘“静止”且在直线行驶时没有转向）。因此，我应用了指定的随机抽样，确保数据在转向角上尽可能平衡。这个过程包括将转向角分成`n`个区间，并为每个区间最多使用`200`个帧：

```python
df = read_csv('data/driving_log.csv')

balanced = pd.DataFrame()   # 平衡数据集
bins = 1000                 # 区间数
bin_n = 200                 # 每个区间包含的样本数（最多）

start = 0
for end in np.linspace(0, 1, num=bins):
    df_range = df[(np.absolute(df.steering) >= start) & (np.absolute(df.steering) < end)]
    range_n = min(bin_n, df_range.shape[0])
    balanced = pd.concat([balanced, df_range.sample(range_n)])
    start = end
balanced.to_csv('data/driving_log_balanced.csv', index=False)
```

结果数据集的直方图在大多数“流行”转向角上看起来相当平衡。

![数据集直方图](https://navoshta.com/images/posts/end-to-end-deep-learning/training_dataset_hist.png)

请注意，我们在平衡数据集时使用的是绝对值，因为在增强过程中应用水平翻转时，我们会为每个帧使用正负转向角。

### 数据增强

平衡约1小时的驾驶数据后，我们得到了**7,698个样本**，这可能不足以让模型很好地泛化。然而，正如许多人指出的，有一些增强技巧可以显著扩展数据集：

- **左、右摄像头**。每个样本附带来自3个摄像头位置的帧：左、中和右。虽然我们只在驾驶时使用中央摄像头，但可以在训练时使用左、右摄像头的数据，并应用转向角修正，将样本数量增加3倍。

```python
cameras = ['left', 'center', 'right']
steering_correction = [.25, 0., -.25]
camera = np.random.randint(len(cameras))
image = mpimg.imread(data[cameras[camera]].values[i])
angle = data.steering.values[i] + steering_correction[camera]
```

- **水平翻转**。对于每个批次，我们将一半的帧水平翻转，并改变转向角的符号，从而将样本数量再增加2倍。

```python
flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
x[flip_indices] = x[flip_indices, :, ::-1, :]
y[flip_indices] = -y[flip_indices]
```

- **垂直移动**。在预处理期间，我们裁剪掉图像中不重要的顶部和底部部分，并随机选择裁剪的量，以提高模型的泛化能力。

```python
top = int(random.uniform(.325, .425) * image.shape[0])
bottom = int(random.uniform(.075, .175) * image.shape[0])
image = image[top:-bottom, :]
```

- **随机阴影**。我们通过降低帧切片的亮度添加一个随机垂直“阴影”，希望使模型对道路上的实际阴影不变。

```python
h, w = image.shape[0], image.shape[1]
[x1, x2] = np.random.choice(w, 2, replace=False)
k = h / (x2 - x1)
b = - k * x1
for i in range(h):
    c = int((i - b) / k)
    image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
```

然后，我们对每个帧进行预处理，裁剪图像的顶部和底部，并调整为模型期望的形状（`32×128×3`，RGB像素强度的32×128图像）。调整大小操作还负责将像素值缩放到`[0, 1]`。

```python
image = skimage.transform.resize(image, (32, 128, 3))
```

为了更好地理解，我们来看一个**单个记录样本**，通过使用所有三个摄像头的帧并应用上述增强管道，将其转换为**16个训练样本**。

![原始帧](https://navoshta.com/images/posts/end-to-end-deep-learning/frames_original.png)

![增强和预处理后的帧](https://navoshta.com/images/posts/end-to-end-deep-learning/frames_augmented.png)

增强管道在`data.py`中使用Keras生成器实现，这让我们可以在CPU上实时进行处理，同时GPU忙于反向传播！

## 模型

我从Nvidia论文中描述的模型开始，并不断简化和优化它，同时确保其在两个赛道上表现良好。显然，我们不需要那么复杂的模型，因为我们处理的数据比Nvidia团队处理的数据简单得多，也更受限制。最终，我确定了一个相当简单的架构，包含**3个卷积层和3个全连接层**。

![模型架构](https://navoshta.com/images/posts/end-to-end-deep-learning/model.png)

这个模型可以用Keras简洁地表示。

```python
from keras import models
from keras.layers import core, convolutional, pooling

model = models.Sequential()
model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(core.Dense(100, activation='relu'))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))
```

我在3个全连接层中的2个添加了dropout，以防止过拟合，模型证明其泛化能力相当不错。模型使用**Adam优化器**进行训练，学习率为`1e-04`，损失函数为**均方误差**。我使用了20%的训练数据作为验证集（这意味着我们只用了**7,698个样本中的6,158个**进行训练），模型在训练约**20个周期**后表现相当不错——你可以在`model.py`中找到与训练相关的代码。

## 结果

汽车在两个赛道上都能几乎无限地顺利驾驶。它很少偏离道路中间，这是在赛道2（之前未见过）上的驾驶情况。

![在之前未见过的赛道上自动驾驶](https://navoshta.com/images/posts/end-to-end-deep-learning/track_2.gif)

你可以查看一个更长的视频，展示汽车在两个赛道上自动驾驶的精彩片段。

显然，这是自动驾驶汽车端到端学习的一个非常基础的例子，但应该能大致了解这些模型的能力，即使考虑到所有在虚拟驾驶模拟器上进行训练和验证的限制。
