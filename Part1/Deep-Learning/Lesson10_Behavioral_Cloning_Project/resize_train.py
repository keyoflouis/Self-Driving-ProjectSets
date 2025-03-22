# import tensorflow as tf
# import os
#
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # 添加到代码开头
#
# # 配置GPU设置（在代码最前面添加）
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # 启用内存自动增长
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(f"{len(gpus)} 物理GPU, {len(logical_gpus)} 逻辑GPU")
#     except RuntimeError as e:
#         print(e)
#
# # 启用混合精度加速（RTX 30系GPU支持）
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)
# print("计算精度策略:", policy.name)
#
# IMAGE_PATH = "data/IMG/"
# CSV_PATH = "data/driving_log.csv"
#
#
# def parse_line(line, correction=0.2):
#     """并行解析CSV行数据"""
#     parts = tf.strings.split(line, ',')
#
#     # 统一处理三个摄像头的路径和转向角
#     image_paths = [tf.strings.strip(parts[i]) for i in [0, 1, 2]]
#     steering_center = tf.strings.to_number(tf.strings.strip(parts[3]), out_type=tf.float32)
#
#     # 生成三个摄像头的转向角
#     steerings = tf.stack([
#         steering_center,  # 中心
#         steering_center + correction,  # 左侧
#         steering_center - correction  # 右侧
#     ])
#
#     return image_paths, steerings
#
#
# def process_image(path):
#     """高效图像处理管道"""
#     # 从路径提取文件名
#     filename = IMAGE_PATH + tf.strings.split(path, '/')[-1]
#
#     # 单次读取解码管道
#     image = tf.io.read_file(filename)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.convert_image_dtype(image, tf.float32)  # 自动归一化到[0,1)
#     image = image - 0.5  # 最终归一化到[-0.5, 0.5)
#     return image
#
#
# def build_dataset(batch_size=64, correction=0.2):
#     """构建高效数据集管道"""
#     # 基础数据集
#     ds = tf.data.TextLineDataset(CSV_PATH).skip(1).cache()
#
#     # 并行解析数据
#     ds = ds.map(
#         lambda line: parse_line(line, correction),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )
#
#     # 展开三个摄像头数据
#     ds = ds.interleave(
#         lambda paths, st: tf.data.Dataset.from_tensor_slices((paths, st)),
#         num_parallel_calls=tf.data.AUTOTUNE,
#         block_length=1
#     )
#
#     # 并行处理图像
#     ds = ds.map(
#         lambda path, st: (process_image(path), st),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )
#
#     # 优化管道
#     return ds.shuffle(1024) \
#         .batch(batch_size) \
#         .prefetch(tf.data.AUTOTUNE)
#
#
# # 使用示例
# dataset = build_dataset(batch_size=128)
#
# # 模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(filters=6, kernel_size=(8, 8), input_shape=(160, 320, 3), padding="SAME", strides=4,
#                            activation="relu", use_bias=True),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#     tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=2, activation="relu", padding="SAME", use_bias=True),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
#
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=128, dtype='float32', use_bias=True,
#                           kernel_initializer=tf.keras.initializers.TruncatedNormal(
#                               mean=0.0,
#                               stddev=0.05,
#                               seed=42
#                           )),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(units=32, dtype='float32', use_bias=True,
#                           kernel_initializer=tf.keras.initializers.TruncatedNormal(
#                               mean=0.0,
#                               stddev=0.05,
#                               seed=42
#                           )),
#     tf.keras.layers.Dense(1)
# ])
#
# model.compile(loss='mse', optimizer='adam')
# model.fit(dataset, epochs=5)
# model.save('model.h5')


import tensorflow as tf
import os

from matplotlib import pyplot as plt

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # 添加到代码开头

# 配置GPU设置（在代码最前面添加）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 启用内存自动增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} 物理GPU, {len(logical_gpus)} 逻辑GPU")
    except RuntimeError as e:
        print(e)

# 启用混合精度加速（RTX 30系GPU支持）
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print("计算精度策略:", policy.name)

IMAGE_PATH = "data/IMG/"
CSV_PATH = "data/driving_log.csv"


def parse_line(line, correction=0.2):
    """并行解析CSV行数据"""
    parts = tf.strings.split(line, ',')

    # 统一处理三个摄像头的路径和转向角
    image_paths = [tf.strings.strip(parts[i]) for i in [0, 1, 2]]
    steering_center = tf.strings.to_number(tf.strings.strip(parts[3]), out_type=tf.float32)

    # 生成三个摄像头的转向角
    steerings = tf.stack([
        steering_center,  # 中心
        steering_center + correction,  # 左侧
        steering_center - correction  # 右侧
    ])

    return image_paths, steerings


def process_image(path):
    """高效图像处理管道"""
    # 从路径提取文件名
    filename = IMAGE_PATH + tf.strings.split(path, '/')[-1]

    # 单次读取解码管道
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # 自动归一化到[0,1)
    image = image - 0.5  # 最终归一化到[-0.5, 0.5)
    return image


def build_dataset(lines_dataset, batch_size=32, correction=0.2, is_training=True):
    """构建高效数据集管道"""
    # 基础数据集
    ds = lines_dataset.cache()

    # 并行解析数据
    ds = ds.map(
        lambda line: parse_line(line, correction),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 展开三个摄像头数据
    ds = ds.interleave(
        lambda paths, st: tf.data.Dataset.from_tensor_slices((paths, st)),
        num_parallel_calls=tf.data.AUTOTUNE,
        block_length=1
    )

    # 并行处理图像
    ds = ds.map(
        lambda path, st: (process_image(path), st),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 训练集需要shuffle，测试集不需要
    if is_training:
        ds = ds.shuffle(1024)

    # 优化管道
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# 创建基础数据集
full_dataset = tf.data.TextLineDataset(CSV_PATH).skip(1)
full_dataset = full_dataset.shuffle(10000, seed=42).cache()  # 初始全局shuffle

# 分割训练集和测试集
total_samples = len(list(full_dataset.as_numpy_iterator()))
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size

print(f"总样本数: {total_samples}, 训练集: {train_size}, 测试集: {test_size}")

# 创建训练和测试数据集
train_lines = full_dataset.take(train_size)
test_lines = full_dataset.skip(train_size)

train_dataset = build_dataset(train_lines, batch_size=32, is_training=True)
test_dataset = build_dataset(test_lines, batch_size=32, is_training=False)

# 模型定义保持不变
model = tf.keras.Sequential([
    tf.keras.layers.Cropping2D(cropping=((70, 25), (0, 0))),
    tf.keras.layers.Resizing(66, 200, interpolation='bilinear'),  # 统一尺寸

    # 标准化层（更适合混合精度）
    tf.keras.layers.LayerNormalization(),

    # 卷积模块
    tf.keras.layers.Conv2D(24, 5, strides=2, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(36, 5, strides=2, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(48, 5, strides=2, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer='l2'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam')

# 训练时只使用训练集
history_object = model.fit(train_dataset, validation_data=test_dataset, epochs=5)

model.save('model.h5')

# 绘制训练集和验证集的损失曲线
plt.plot(history_object.history['loss'])          # 训练损失
plt.plot(history_object.history['val_loss'])      # 验证损失
plt.title('Model Mean Squared Error Loss')        # 标题
plt.ylabel('Mean Squared Error Loss')             # Y轴标签
plt.xlabel('Epoch')                               # X轴标签
plt.legend(['Training Set', 'Validation Set'], loc='upper right')  # 图例
plt.show()