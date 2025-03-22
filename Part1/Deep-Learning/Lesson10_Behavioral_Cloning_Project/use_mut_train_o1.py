import tensorflow as tf
import os
import numpy as np
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

IMG_PATH = "data/IMG/"
CSV_PATH = "data/driving_log.csv"


def parse_line(line, correction):
    """ 并行解析CSV行数据 """

    part = tf.strings.split(line, ',')

    image_path = [tf.strings.strip(part[i]) for i in [0, 1, 2]]
    steering_center = tf.strings.to_number(tf.strings.strip(part[3]), out_type=tf.float32)

    steerings = tf.stack([
        steering_center,
        steering_center + correction,
        steering_center - correction
    ])

    return image_path, steerings


def process_image(path):
    filename = IMG_PATH + tf.strings.split(path, '/')[-1]

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32) - 0.5
    return image


def augment_data(image, steering):
    # 确保steering是标量
    steering = tf.reshape(steering, [])

    # 数据增强
    flipped_image = tf.image.flip_left_right(image)
    flipped_steering = -steering

    # 创建包含两个样本的数据集
    return tf.data.Dataset.from_tensors(
        (image, steering)
    ).concatenate(
        tf.data.Dataset.from_tensors((flipped_image, flipped_steering))
    )


def build_dataset(line_dataset, batch_size=32, correction=0.2, is_training=True):
    ds = line_dataset.cache()

    # 并行读取图像
    ds = ds.map(
        lambda line: parse_line(line, correction),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 并行处理图像
    ds = ds.interleave(
        lambda paths, st: tf.data.Dataset.from_tensor_slices((paths, st)),
        num_parallel_calls=tf.data.AUTOTUNE,
        block_length=1
    )

    ds = ds.map(
        lambda path, st: (process_image(path), st),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.flat_map(
        lambda image,st :augment_data(image,st),
    )

    if is_training:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



# 载入与划分训练集
full_dataset = tf.data.TextLineDataset(CSV_PATH).skip(1)
full_dataset = full_dataset.shuffle(10000, seed=42).cache()
total_samples = len(list(full_dataset.as_numpy_iterator()))
train_size = int(total_samples * 0.8)
test_size = total_samples - train_size
print(f"总样本数: {total_samples}, 训练集: {train_size}, 测试集: {test_size}")
train_lines = full_dataset.take(train_size)
test_lines = full_dataset.skip(train_size)


# 数据集构建
train_dataset = build_dataset(train_lines, batch_size=32, is_training=True)
test_dataset = build_dataset(test_lines, batch_size=32, is_training=False)


# 绘制转向角分布直方图
steering_angles = []
for sample in train_dataset.unbatch():
    steering_angles.append(sample[1].numpy())
plt.hist(steering_angles, bins=50)
plt.title("Steering Angle Distribution")
plt.show()


# 收集转向角数据
steering_angles = []
for sample in train_dataset.unbatch():
    steering_angles.append(sample[1].numpy())
steering_angles = np.array(steering_angles)

# 使用numpy计算直方图数据（无需绘图）
hist_counts, bin_edges = np.histogram(steering_angles, bins=50)

# 控制台打印直方图数据
print("直方图区间分布 [左边界, 右边界) : 样本数量")
for i in range(len(hist_counts)):
    left = bin_edges[i]
    right = bin_edges[i+1]
    count = hist_counts[i]
    print(f"[{left:.4f}, {right:.4f}) : {int(count)}")

# 可选：打印统计摘要
print("\n统计摘要:")
print(f"总样本数: {len(steering_angles)}")
print(f"直方图区间宽度: {(bin_edges[1]-bin_edges[0]):.4f}")
print(f"最小转向角: {np.min(steering_angles):.4f}")
print(f"最大转向角: {np.max(steering_angles):.4f}")


# 模型定义保持不变
model = tf.keras.Sequential([
    tf.keras.layers.Cropping2D(cropping=((60, 25), (0, 0))),
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
history_object = model.fit(train_dataset, validation_data=test_dataset, epochs=30)


model.save('model.h5')

# # 绘制训练集和验证集的损失曲线
# plt.plot(history_object.history['loss'])          # 训练损失
# plt.plot(history_object.history['val_loss'])      # 验证损失
# plt.title('Model Mean Squared Error Loss')        # 标题
# plt.ylabel('Mean Squared Error Loss')             # Y轴标签
# plt.xlabel('Epoch')                               # X轴标签
# plt.legend(['Training Set', 'Validation Set'], loc='upper right')  # 图例
# plt.show()