import tensorflow as tf
import numpy as np

# 加载MNIST数据集
(X_test, Y_test), (_, _) = tf.keras.datasets.mnist.load_data()

# 预处理数据
X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

X_test = np.pad(X_test,((0,0),(2,2),(2,2),(0,0)),'constant')

assert(len(X_test)==len(Y_test))

# 只使用最后5000个样本进行测试
X_test = X_test[-5000:]
Y_test = Y_test[-5000:]

# 加载预训练的LeNet模型
model = tf.keras.models.load_model('./lenet_model')

# 评估模型
loss, acc = model.evaluate(X_test, Y_test)

# 打印准确率
print(f"Test Accuracy: {acc:.4f}")