
import tensorflow as tf
import pickle



def load_bottleneck_data(training_file,validation_file):
    """
        加载瓶颈特征的实用函数。

        参数:
            training_file - 字符串
            validation_file - 字符串
        """
    print("训练文件", training_file)
    print("验证文件", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val

def main(training_file,validation_file):
    X_train ,Y_train, X_Val,Y_val =load_bottleneck_data(training_file,validation_file)

    X_train=X_train.reshape(-1,512)
    X_Val = X_Val.reshape(-1,512)
    print(X_Val.shape)
    print(X_train.shape,Y_train.shape)



    # ToDO:定义模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(input_shape=((512,)),
                              activation="relu",
                              use_bias=True,
                              units=128,
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(activation="softmax",units=43)
    ])


    # ToDO：训练模型
    model.compile(optimizer="adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              epochs =50,
              validation_data=(X_Val,Y_val),
              batch_size=128)


if __name__=="__main__":
    training_file_1 = "vgg-100/vgg_cifar10_100_bottleneck_features_train.p"
    training_file_2 = "vgg-100/vgg_traffic_100_bottleneck_features_train.p"

    validation_file1 = "vgg-100/vgg_cifar10_bottleneck_features_validation.p"
    validation_file2 = "vgg-100/vgg_traffic_bottleneck_features_validation.p"

    main(training_file_2,validation_file2)