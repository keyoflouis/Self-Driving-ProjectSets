import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

def preprocess(images,labels):

    features =images.reshape((-1,784)).astype(np.float32)/255.0
    # 转化标签为相应的 one-hote 码
    labels =tf.keras.utils.to_categorical(labels,10).astype(np.float32)

    return features,labels


train_features,train_labels = preprocess(train_images,train_labels)
test_features,test_labels =preprocess(test_images,test_labels)

learning_rate = 0.05
batch_size = 128
n_input =784
n_class=10
epochs =20

weight =tf.Variable(tf.random.normal([n_input,n_class],dtype=tf.float32))
bias =tf.Variable(tf.random.normal(shape=[n_class],dtype=tf.float32))

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

#  创建数据集
train_datasets =tf.data.Dataset.from_tensor_slices((train_features,train_labels)).shuffle(60000).batch(batch_size)

for epoch in range(epochs):

    for batch_features ,batch_labels in train_datasets:

        with tf.GradientTape() as tape:

            logits =tf.matmul(batch_features,weight)+bias

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_labels,logits=logits))

        gradients =tape.gradient(loss,[weight,bias])

        optimizer.apply_gradients(zip(gradients,[weight,bias]))

test_logits =tf.matmul(test_features,weight)+bias

correct_pred =tf.equal(tf.argmax(test_logits,1),tf.argmax(test_labels,1))
accuracy =tf.reduce_mean(tf.cast(correct_pred,tf.float32))

print(format(accuracy.numpy()))















