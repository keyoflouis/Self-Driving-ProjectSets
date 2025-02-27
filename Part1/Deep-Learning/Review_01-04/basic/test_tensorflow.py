import tensorflow as tf
import numpy as np

# tf.data 有什么功能
### 创建数据集，预处理数据集。比如在SDG中实现mini-batch


# 单变量数据集，即分析和预测这个单一变量的未来值。
data = np.array([1, 2, 3, 4, 5])

data_set = tf.data.Dataset.from_tensor_slices(data)

data_set =data_set.map(lambda x:x*2)

data_set.batch(2)



# 假设你有一个数据集
dataset = tf.data.Dataset.range(10)  # 创建一个包含0到9的数据集

# 将数据集中的元素组合成批次，每个批次包含2个元素
dataset = dataset.batch(2)

for i in dataset :
    print(i.numpy())

# 迭代数据集并访问每个批次
#for batch in dataset:
#    print(batch.numpy())  # 将Tensor转换为NumPy数组并打印



