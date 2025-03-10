import numpy as np
import tensorflow as tf

net_data = np.load("bvlc-alexnet.npy", encoding="latin1", allow_pickle=True).item()


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    c_i = input.shape[-1]  # 修改点1: input.shape() -> input.shape
    assert c_i % group == 0
    assert c_o % group == 0

    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, axis=3)
        kernel_groups = tf.split(kernel, group, axis=3)  # 修改点2: 分割轴由2改为3
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, axis=3)

    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def AlexNet(features, feature_extract=False):
    # conv1
    k_h, k_w, c_o, s_h, s_w = 11, 11, 96, 4, 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(features, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

    # maxpool1
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # conv2
    k_h, k_w, c_o, s_h, s_w, group = 5, 5, 256, 1, 1, 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

    # maxpool2
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # conv3
    k_h, k_w, c_o, s_h, s_w = 3, 3, 384, 1, 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    k_h, k_w, c_o, s_h, s_w, group = 3, 3, 384, 1, 1, 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    k_h, k_w, c_o, s_h, s_w, group = 3, 3, 256, 1, 1, 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool5
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # fc6
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    reshaped_fc6 = tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.shape[1:]))])  # 修改点3: get_shape()[1:] -> shape[1:]
    fc6 = tf.nn.relu(tf.matmul(reshaped_fc6, fc6W) + fc6b)  # 替换relu_layer

    # fc7
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu(tf.matmul(fc6, fc7W) + fc7b)  # 替换relu_layer

    if feature_extract:
        return fc7

    # fc8
    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    logits = tf.matmul(fc7, fc8W) + fc8b  # 替换xw_plus_b
    probabilities = tf.nn.softmax(logits)

    return probabilities


# # 代码有错误
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, Model
#
#
# def load_weights():
#    return np.load("bvlc-alexnet.npy", encoding="latin1", allow_pickle=True).item()
#
#
# class AlexNet(Model):
#    def __init__(self, feature_extract=False, num_classes=1000):
#        super(AlexNet, self).__init__()
#        self.feature_extract = feature_extract
#        net_data = load_weights()
#
#        # 打印权重形状以调试
#        print("[DEBUG] fc6 weights shape:", net_data['fc6'][0].shape)
#        print("[DEBUG] fc6 weights after transpose:", net_data['fc6'][0].T.shape)
#
#        # Conv1
#        self.conv1 = layers.Conv2D(
#            filters=96, kernel_size=(11, 11), strides=4, padding='same',
#            activation='relu', name='conv1')
#        self.conv1.build((None, None, None, 3))
#        self.conv1.set_weights([net_data['conv1'][0], net_data['conv1'][1]])
#
#        # LRN1 & MaxPool1
#        self.lrn1 = layers.Lambda(
#            lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0))
#        self.pool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding='valid')
#
#        # Conv2（分组卷积）
#        self.conv2 = layers.Conv2D(
#            filters=256, kernel_size=(5, 5), padding='same', activation='relu',
#            groups=2, name='conv2')
#        self.conv2.build((None, None, None, 96))
#        self.conv2.set_weights([net_data['conv2'][0], net_data['conv2'][1]])
#
#        # LRN2 & MaxPool2
#        self.lrn2 = layers.Lambda(
#            lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0))
#        self.pool2 = layers.MaxPooling2D(pool_size=3, strides=2, padding='valid')
#
#        # Conv3
#        self.conv3 = layers.Conv2D(
#            filters=384, kernel_size=(3, 3), padding='same', activation='relu', name='conv3')
#        self.conv3.build((None, None, None, 256))
#        self.conv3.set_weights([net_data['conv3'][0], net_data['conv3'][1]])
#
#        # Conv4（分组卷积）
#        self.conv4 = layers.Conv2D(
#            filters=384, kernel_size=(3, 3), padding='same', activation='relu',
#            groups=2, name='conv4')
#        self.conv4.build((None, None, None, 384))
#        self.conv4.set_weights([net_data['conv4'][0], net_data['conv4'][1]])
#
#        # Conv5（分组卷积）
#        self.conv5 = layers.Conv2D(
#            filters=256, kernel_size=(3, 3), padding='same', activation='relu',
#            groups=2, name='conv5')
#        self.conv5.build((None, None, None, 384))
#        self.conv5.set_weights([net_data['conv5'][0], net_data['conv5'][1]])
#
#        # MaxPool5
#        self.pool5 = layers.MaxPooling2D(pool_size=3, strides=2, padding='valid')
#
#        # FC Layers
#        self.flatten = layers.Flatten()
#
#        # FC6（修复权重转置）
#        self.fc6 = layers.Dense(4096, activation='relu', name='fc6')
#        self.fc6.build((None, 6 * 6 * 256))  # 输入维度9216
#        # 根据实际权重形状决定是否需要转置
#        if net_data['fc6'][0].shape == (4096, 9216):
#            self.fc6.set_weights([net_data['fc6'][0].T, net_data['fc6'][1]])  # 转置权重
#        else:
#            self.fc6.set_weights([net_data['fc6'][0], net_data['fc6'][1]])  # 直接使用
#
#        # FC7
#        self.fc7 = layers.Dense(4096, activation='relu', name='fc7')
#        self.fc7.build((None, 4096))
#        if net_data['fc7'][0].shape == (4096, 4096):
#            self.fc7.set_weights([net_data['fc7'][0].T, net_data['fc7'][1]])
#        else:
#            self.fc7.set_weights([net_data['fc7'][0], net_data['fc7'][1]])
#
#        # FC8
#        if not feature_extract:
#            self.fc8 = layers.Dense(num_classes, name='fc8')
#            self.fc8.build((None, 4096))
#            if net_data['fc8'][0].shape == (num_classes, 4096):
#                self.fc8.set_weights([net_data['fc8'][0].T, net_data['fc8'][1]])
#            else:
#                self.fc8.set_weights([net_data['fc8'][0], net_data['fc8'][1]])
#
#    def call(self, inputs):
#        x = self.conv1(inputs)
#        x = self.lrn1(x)
#        x = self.pool1(x)
#
#        x = self.conv2(x)
#        x = self.lrn2(x)
#        x = self.pool2(x)
#
#        x = self.conv3(x)
#        x = self.conv4(x)
#        x = self.conv5(x)
#        x = self.pool5(x)
#
#        x = self.flatten(x)
#        x = self.fc6(x)
#        x = self.fc7(x)
#
#        if self.feature_extract:
#            return x
#
#        x = self.fc8(x)
#        return tf.nn.softmax(x)
#
#
# # 使用示例
# model = AlexNet(feature_extract=False)
# input_tensor = tf.random.normal((1, 227, 227, 3))  # AlexNet标准输入尺寸
# output = model(input_tensor)
# print(output.shape)
