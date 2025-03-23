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
import numpy as np
import tensorflow as tf
import os

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

# 收集转向角数据
steering_angles = []
for sample in train_dataset.unbatch():
    steering_angles.append(sample[1].numpy())
steering_angles = np.array(steering_angles)

# 统计0值样本数量
count_zero = np.sum(steering_angles == 0)

# 使用numpy计算直方图数据（保持区间总数不变）
hist_counts, bin_edges = np.histogram(
    steering_angles,
    bins=50,
    range=(-1.0, 1.0)
)

# 找到包含0值的区间索引
zero_bin_index = None
for i in range(len(bin_edges) - 1):
    if bin_edges[i] <= 0 < bin_edges[i + 1]:
        zero_bin_index = i
        break

# 调整直方图计数：从原区间扣除0值样本
if zero_bin_index is not None and count_zero > 0:
    hist_counts[zero_bin_index] -= count_zero

# 打印数据分布
print("直方图区间分布 [左边界, 右边界) : 样本数量 占比")
print(f"{'转向区间':<25} | {'样本数量':<8} | {'占比 (%)':<6}")
print("-" * 45)

total_samples = len(steering_angles)
for i in range(len(hist_counts)):
    left = bin_edges[i]
    right = bin_edges[i + 1]
    count = hist_counts[i]
    percent = (count / total_samples) * 100

    # 特殊处理包含0值的区间
    if i == zero_bin_index and count_zero > 0:
        # 打印非零部分
        range_str = f"[{left:.4f}, {right:.4f})"
        if abs(left) < 0.1:
            range_str += " (直行)"
        elif left > 0.3:
            range_str += " (急右转)"
        elif right < -0.3:
            range_str += " (急左转)"

        print(f"{range_str:<25} | {count:<8} | {percent:.2f}%")

        # 单独打印0值区间
        zero_percent = (count_zero / total_samples) * 100
        print(f"[0.0000, 0.0000) (直行)       | {count_zero:<8} | {zero_percent:.2f}%")
    else:
        # 正常打印其他区间
        range_str = f"[{left:.4f}, {right:.4f})"
        if abs(left) < 0.1:
            range_str += " (直行)"
        elif left > 0.3:
            range_str += " (急右转)"
        elif right < -0.3:
            range_str += " (急左转)"

        print(f"{range_str:<25} | {count:<8} | {percent:.2f}%")

# 扩展统计摘要（保持原样）
print("\n关键统计：")
print(f"总样本数: {total_samples}")
print(f"中位数: {np.median(steering_angles):.4f}")
print(f"均值: {np.mean(steering_angles):.4f}")
print(f"标准差: {np.std(steering_angles):.4f}")
print(f"最小转向角: {np.min(steering_angles):.4f}")
print(f"最大转向角: {np.max(steering_angles):.4f}")
print(f"|θ| < 0.1 样本占比: {np.mean(np.abs(steering_angles) < 0.1) * 100:.2f}%")
print(f"|θ| > 0.4 样本占比: {np.mean(np.abs(steering_angles) > 0.4) * 100:.2f}%")


# 模型定义保持不变
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(8, 8), input_shape=(160, 320, 3), padding="SAME", strides=4,
                           activation="relu", use_bias=True),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=2, activation="relu", padding="SAME", use_bias=True),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, dtype='float32', use_bias=True,
                          kernel_initializer=tf.keras.initializers.TruncatedNormal(
                              mean=0.0,
                              stddev=0.05,
                              seed=42
                          )),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=32, dtype='float32', use_bias=True,
                          kernel_initializer=tf.keras.initializers.TruncatedNormal(
                              mean=0.0,
                              stddev=0.05,
                              seed=42
                          )),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam')

# 训练时只使用训练集
history = model.fit(train_dataset,validation_data=test_dataset, epochs=5)


model.save('model.h5')