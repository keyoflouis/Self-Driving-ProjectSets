# TODO:
#  1.测试不会sanbox.py中不会的语法
#  2.尝试写一遍sanbox.py训练过程，包括数据预处理


import numpy as np
import tensorflow as tf
def get_weights(n_features,n_labels):
    return tf.Variable(tf.random.normal([n_features,n_labels],stddev=0.1))

def get_biases(n_labels):
    return tf.Variable(tf.zeros([n_labels]))

def linear(x,w,b):
    return tf.matmul(x,w)+b



def mnist_feature_label(n_labels):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # reshape()的参数，-1和748改动会如何，他的参数可以有那些
    train_images = train_images.reshape((-1, 784)).astype('float32') / 255.0

    # 这里的to_categorical具体是什么，为什么能将train_labels变为one-hot，他的参数可以有哪些
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)

    # 练习arange,shuffle是什么，可以有哪些参数
    indices = np.arange(len(train_images))
    np.random.shuffle(indices)

    # 这里的sampled_indices是随机打乱后的前1万个indices的元素（记录的值代表索引）组成的数组么
    sampled_indices = indices[:10000]
    sampled_images = train_images[sampled_indices]
    sampled_labels = train_labels[sampled_indices]

    # 取所有lable的前三个元素的any，得到一个和label同大小的mask数组，用于过滤出0,1,2这三个数字
    mask = np.any(sampled_labels[:, :n_labels], axis=1)
    filtered_images = sampled_images[mask]
    filtered_labels = sampled_labels[mask][:, :n_labels]

    return filtered_images, filtered_labels


n_features = 784
n_labels = 3

train_features, train_labels = mnist_feature_label(n_labels)

features = tf.convert_to_tensor(train_features, dtype=tf.float32)
labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)

w =get_weights(n_features,n_labels)

b =get_biases(n_labels)


