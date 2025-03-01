import tensorflow as tf
from tensorflow.keras.datasets import mnist

((train_img,train_label),(test_img,test_label))=mnist.load_data()
model=tf.keras.models.load_model("my_model")

prediction=model.predict(test_img)

for i in range(10):
    pre = tf.argmax(prediction[i])
    print(f'pre:{pre},label:{test_label[i]}')