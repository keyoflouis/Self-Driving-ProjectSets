from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import tensorflow as tf

(x_train , y_train ),(x_test,y_test) =cifar10.load_data()

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.02, stratify=y_train, random_state=42)

def normalize(image_data):
    return image_data / 255.0

x_train =normalize(x_train)
x_test = normalize(x_test)
x_val =normalize(x_val)


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
        tf.keras.layers.Dense(units=10)]

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
          x=x_train,
          y=y_train,
          validation_data=(x_test, y_test))
loss, acc = model.evaluate(x=x_val, y=y_val)

model.save("./model_data")

print(format(loss))
print(format(acc))
