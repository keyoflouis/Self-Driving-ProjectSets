import tensorflow as tf
from tensorflow.keras import datasets,layers,Model

(x_train,y_train),(x_test,y_test) =datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.0
y_train =tf.keras.utils.to_categorical(y_train,10)
y_test =tf.keras.utils.to_categorical(y_test,10)

x_val =x_train[-5000:]
y_val =y_train[-5000:]
x_train =x_train[:-5000]
y_train =y_train[:-5000]


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(128)

class CNN(Model):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 =layers.Conv2D(32,(5,5),padding='same',activation='relu')
        self.pool1 =layers.MaxPooling2D((2,2),strides=2,padding='same')
        self.conv2 =layers.Conv2D(64,(5,5),padding='same',activation='relu')
        self.pool2 =layers.MaxPooling2D((2,2),strides=2,padding='same')
        self.flatten =layers.Flatten()
        self.fc1 =layers.Dense(1024,activation='relu')
        self.dropout =layers.Dropout(0.25)
        self.out =layers.Dense(10,activation='softmax')

    # 前向传播，forward
    def call(self,x,training=False):
        x =self.conv1(x)
        x =self.pool1(x)
        x =self.conv2(x)
        x =self.pool2(x)
        x =self.flatten(x)
        x =self.fc1(x)
        if training:
            x =self.dropout(x,trainng =training)
        return self.out(x)


learning_rate =0.05

model =CNN()
optimizer =tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss_fn =tf.keras.losses.CategoricalCrossentropy()

epochs = 10
test_valid_size = 256

for epoch in range(epochs):
    for batch_idx,(image,labels) in enumerate(train_dataset):

        with tf.GradientTape() as tp:
            logits = model(image,training =True)
            loss =loss_fn(labels,logits)

        gradients =tp.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))

        val_logits =model(x_val[:test_valid_size],training=False)
        val_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(val_logits,1),tf.argmax(y_val[:test_valid_size],1)),tf.float32)
        )

        if batch_idx %100 ==0:
            print(f'epoch {epoch+1:>2},Batch{batch_idx+1:>3} -'
                  f'Loss :{loss.numpy():>10.4f} Validation Accuracy:{val_acc.numpy():.6f}')



test_logits =model(x_test[:test_valid_size],training=False)
test_acc =tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(test_logits, 1), tf.argmax(y_test[:test_valid_size], 1)), tf.float32)
)

print(f'\nTesting Accuracy: {test_acc.numpy():.6f}')












