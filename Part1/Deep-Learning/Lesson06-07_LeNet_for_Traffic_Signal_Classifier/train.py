import pickle
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# 数据载入
train_file = 'traffic-signs-data/train.p'
test_file = 'traffic-signs-data/test.p'
with open(train_file, mode='rb') as f:
    train = pickle.load(f)
with open(test_file, mode='rb') as f:
    test = pickle.load(f)

X_train, Y_train = train["features"], train["labels"]
X_test, Y_test = test["features"], test["labels"]

n_train = len(X_train)
n_test = len(X_test)
image_shape = X_train[0].shape
n_class = len(set(Y_test))

print("训练样本数 =", n_train)
print("测试样本数 =", n_test)
print("图像尺寸 =", image_shape)
print("类别数 =", n_class)

# 随机打印
index = random.randint(0, len(X_train))
plt.imshow(X_train[index].squeeze(), cmap="gray")
plt.show()
print("标签：", Y_train[index])


# 归一化
def normalize(image_data):
    return 0.01 + (image_data / 255.0) * 0.98

X_train = normalize(image_data=X_train)
X_test = normalize(image_data=X_test)
X_train, Y_train = shuffle(X_train, Y_train)


# 划分验证集
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.02, random_state=42)


# 定义模型
def LeNet():
    model = tf.keras.models.Sequential([
        # s=1,p=1,k=7,output=28x28x6
        tf.keras.layers.Conv2D(input_shape=(32, 32, 3), filters=6, kernel_size=(5, 5), padding="valid", strides=1,
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

        # s=1,p=0,k=4,output=10x10x16
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid", strides=1,
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(strides=2),

        # Flatten 展平非批量维度
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=120, activation='relu'),
        tf.keras.layers.Dense(units=84, activation='relu'),
        tf.keras.layers.Dense(units=43)]

    )
    return model


model = LeNet()

model.compile(optimizer='adam',
              metrics=['accuracy'],
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

epochs = 50
batch_size = 256
model.fit(epochs=epochs,
          batch_size=batch_size,
          x=X_train,
          y=Y_train,
          validation_data=(X_test, Y_test))
loss, acc = model.evaluate(x=X_val, y=Y_val)

model.save("./model_data")

print(format(loss))
print(format(acc))
