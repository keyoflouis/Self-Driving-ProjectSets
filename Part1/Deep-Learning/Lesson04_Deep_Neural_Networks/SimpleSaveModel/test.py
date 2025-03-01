import tensorflow as tf
from tensorflow.keras.datasets import mnist

((train_img,train_label),(test_img,test_label))=mnist.load_data()

train_img = train_img/255.0
test_img = test_img/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_img,train_label,epochs=5,validation_data=(test_img,test_label))

model.save('my_model')