import tensorflow as tf
import  numpy as np
from numpy.array_api import astype

(train_img,train_label),(test_img,test_label) = tf.keras.datasets.mnist.load_data()

train_img = train_img/255.0
test_img = test_img/255.0

train_label =tf.keras.utils.to_categorical(train_label,10).astype(np.float32)
test_label =tf.keras.utils.to_categorical(test_label,10,).astype(np.float32)

train_img = train_img.reshape(-1,28*28).astype(np.float32)
test_img =test_img.reshape(-1,28*28).astype(np.float32)

input_shape = 784
class_n = 10
learning_rate =0.1
batch_size =128
epoches =5

Weights =tf.Variable(tf.random.normal([input_shape,class_n],dtype=tf.float32))
Bias = tf.Variable(tf.random.normal([class_n],dtype=tf.float32))

optimizer =tf.keras.optimizers.SGD(learning_rate=learning_rate)

train_dataset =tf.data.Dataset.from_tensor_slices((train_img,train_label)).shuffle(60000).batch(batch_size)

for epoch in range(epoches):
    for batch_feature,batch_label in train_dataset:
        with tf.GradientTape() as tp:
            logits= tf.matmul(batch_feature,Weights) + Bias

            # 平均值
            loss = tf.reduce_mean(
                # 用logits计算softmax得到预测概率，然后用得到的预测概率计算交叉熵，
                # 用真实标签的one-hot码乘以对应位的交叉熵，求和得到了训练某一样本时的交叉熵。
                tf.nn.softmax_cross_entropy_with_logits(
                labels=batch_label,logits=logits
                )
            )

        gradient = tp.gradient(loss,[Weights,Bias])
        optimizer.apply_gradients(zip(gradient,[Weights,Bias]))


# 测试准确率
test_logits = tf.matmul(test_img, Weights) + Bias
correct_pred = tf.equal(tf.argmax(test_logits, 1), tf.argmax(test_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('Test Accuracy: {:.3f}'.format(accuracy.numpy()))













