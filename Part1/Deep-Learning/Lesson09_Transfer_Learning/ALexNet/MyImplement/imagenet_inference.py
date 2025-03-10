# NOTE: Minimal changes for TensorFlow 2.x compatibility
import time
import numpy as np
import tensorflow.compat.v1 as tf  # 修改点1: 使用兼容模式

tf.disable_v2_behavior()
import imageio.v2 as imageio  # 修改点2: 替换scipy的imread
from caffe_classes import class_names
from alexnet import AlexNet

# 创建占位符（保持TensorFlow 1.x语法）
x = tf.placeholder(tf.float32, (None, 227, 227, 3))

# 构建AlexNet（需确保alexnet.py已适配TF2.x）
probs = AlexNet(x, feature_extract=False)

# 初始化操作
init = tf.global_variables_initializer()  # 修改点3: 更新初始化方法
sess = tf.Session()
sess.run(init)

# 读取图片（使用imageio代替scipy）
im1 = imageio.imread("poodle.png")[:, :, :3].astype(np.float32)  # 修改点4
im1 = im1 - np.mean(im1)

im2 = imageio.imread("weasel.png")[:, :, :3].astype(np.float32)  # 修改点4
im2 = im2 - np.mean(im2)

# 运行推理（保持TensorFlow 1.x会话方式）
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})


# 打印结果（保持不变）
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))



# # 有错误
# import imageio.v2 as imageio
# import numpy as np
# from skimage.transform import resize
# from alexnet import AlexNet
# from caffe_classes import class_names
# import time
#
# # 读取图像并确保通道数为3
# x_1 = imageio.imread("poodle.png")[:, :, :3]
# x_2 = imageio.imread("weasel.png")[:, :, :3]
#
# # 调整图像大小为 (227, 227, 3)
# x_1 = resize(x_1, (227, 227))
# x_2 = resize(x_2, (227, 227))
#
# # 归一化图像
# x_1 = x_1 - np.mean(x_1)
# x_2 = x_2 - np.mean(x_2)
#
# # 转换为 NumPy 数组
# x_1 = np.array(x_1)
# x_2 = np.array(x_2)
#
# # 初始化模型（假设加载了预训练权重）
# model = AlexNet(feature_extract=False)
#
# # 添加批次维度并预测
# result1 = model.predict(np.expand_dims(x_1, axis=0))
# result2 = model.predict(np.expand_dims(x_2, axis=0))
#
# class_r1 = np.argmax(result1)
# class_r2 = np.argmax(result2)
#
# print("Predicted class for poodle.png:",class_names[class_r1],np.max(result1))
# print("Predicted class for weasel.png:",class_names[class_r2],np.max(result2))
#
