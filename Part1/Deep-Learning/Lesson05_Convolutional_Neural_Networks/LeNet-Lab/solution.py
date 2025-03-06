import random

from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf

# 加载，处理数据
(X_train ,Y_train),(X_test,Y_test) = mnist.load_data()
X_train=X_train.reshape((-1,28,28,1)).astype('float32')/255.0
X_test=X_test.reshape((-1,28,28,1)).astype('float32')/255.0

X_val = X_train[55000:]
Y_val = Y_train[55000:]
X_train =X_train[:55000]
Y_train =Y_train[:55000]

# padding
X_train=np.pad(X_train,((0,0),(2,2),(2,2),(0,0)),'constant')
X_val = np.pad(X_val,((0,0),(2,2),(2,2),(0,0)),'constant')
X_test =np.pad(X_test,((0,0),(2,2),(2,2),(0,0)),'constant')

# 断言检查
assert(len(X_train) == len(Y_train))
assert(len(X_val) == len(Y_val))
assert(len(X_test) == len(Y_test))

print("\n图像形状: {}".format(X_train[0].shape))
print("训练集:   {} 样本".format(len(X_train)))
print("验证集: {} 样本".format(len(X_val)))
print("测试集:       {} 样本".format(len(X_test)))

# 可视化数据
index = random.randint(0, len(X_train)-1)
image = X_train[index].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
plt.show()
print("样本标签:", Y_train[index])


# TODO 构建LeNet模型
def LeNet():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6,(5,5),activation="relu",input_shape=(32,32,1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2),
        tf.keras.layers.Conv2D(16,(5,5),activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120,activation='relu'),
        tf.keras.layers.Dense(84,activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

model =LeNet()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs =10
batch_size =128

history = model.fit(X_train,Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val,Y_val),
                    verbose=1)
model.save('lenet_model')

test_loss, test_acc =model.evaluate(X_test,Y_test,verbose=0)
print("\n测试精度 = {:.3f}".format(test_acc))