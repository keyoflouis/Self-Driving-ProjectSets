from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
import os
import math

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# 配置GPU设置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} 物理GPU, {len(logical_gpus)} 逻辑GPU")
    except RuntimeError as e:
        print(e)

# 启用混合精度加速
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print("计算精度策略:", policy.name)

IMAGE_PATH = "data/IMG/"
CSV_PATH = "data/driving_log_balanced.csv"

# 读取数据集
full_dataset = tf.data.TextLineDataset(CSV_PATH).skip(1)
count = full_dataset.reduce(0, lambda x, _: x + 1).numpy()
full_dataset = full_dataset.shuffle(1024)
total_train = int(count * 0.8)

train_dataset = full_dataset.take(total_train)
test_dataset = full_dataset.skip(total_train)


def build_datasets(line_datasets, is_training=True):
    """构建数据管道"""

    def parse_line(line):
        """读取每一行CSV，返回图像路径和转向角度"""
        line = tf.strings.split(line, ",")
        steering = tf.strings.to_number(line[3], out_type=tf.float32)

        paths = tf.stack([
            tf.strings.strip(line[0]),
            tf.strings.strip(line[1]),
            tf.strings.strip(line[2])
        ])
        steers = tf.stack([steering, steering + 0.15, steering - 0.15])
        steers = tf.clip_by_value(steers, -1.0, 1.0)

        return paths, steers

    def read_process_image(image_path):
        """读取图片并进行预处理"""
        file = tf.io.read_file(IMAGE_PATH + tf.strings.split(image_path, "/")[-1])
        image_file = tf.io.decode_jpeg(file, channels=3)
        image_file = tf.image.resize(image_file, [160, 320])
        return tf.image.convert_image_dtype(image_file, dtype=tf.float32) - 0.5

    def augment_data(image, steering):
        """极端样本反转+随机光亮，普通样本反转"""
        ds = tf.data.Dataset.from_tensors((image, steering))
        flipped_image = tf.image.flip_left_right(image)
        flipped_label = -steering
        ds.concatenate(tf.data.Dataset.from_tensors((flipped_image, flipped_label)))

        abs_steer = tf.abs(steering)
        ds = tf.cond(
            abs_steer > 0.4,
            lambda:
            (ds.concatenate(tf.data.Dataset.from_tensors((tf.image.random_brightness(image, max_delta=0.3), steering))))
            .concatenate(
                tf.data.Dataset.from_tensors((tf.image.random_brightness(flipped_image, max_delta=0.3), -steering))),
            lambda: ds
        )

        return ds

    def resample_data(image, steering):
        """根据转向角度重采样数据"""
        abs_steer = tf.abs(steering)

        repeat_times = tf.case([
            (tf.greater_equal(abs_steer, 0.4), lambda: tf.constant(3, tf.int64)),  # 从2提高到5
            (tf.greater_equal(abs_steer, 0.2), lambda: tf.constant(2, tf.int64))  # 新增中等转向增强
        ], default=lambda: tf.constant(1, tf.int64))

        return tf.data.Dataset.from_tensors((image, steering)).repeat(repeat_times)

    def desample_data_0(image, steering):
        """降采样直行数据"""
        should_downsample = tf.logical_and(
            tf.abs(steering) == 0,
            tf.random.uniform(shape=[], seed=42) < 0.25
        )
        return tf.logical_not(should_downsample)

    def desample_data_0_12_0_16(image, steering):
        should_downsample = tf.logical_and(
            tf.logical_and(tf.abs(steering) < 0.16, tf.abs(steering) > 0.12),
            tf.random.uniform(shape=[], seed=42) < 0.25
        )
        return tf.logical_not(should_downsample)

    dataset = line_datasets.cache()

    # 解析CSV行数据
    dataset = dataset.map(
        lambda line: parse_line(line),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 展开数据
    dataset = dataset.interleave(
        lambda paths, steers: tf.data.Dataset.from_tensor_slices((paths, steers)),
        num_parallel_calls=tf.data.AUTOTUNE,
        block_length=1
    )

    # 读取并预处理图片
    dataset = dataset.map(
        lambda path, steering: (read_process_image(path), steering),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_training:
        # 数据增强/重采样
        # dataset = dataset.flat_map(augment_data)
        # dataset = dataset.flat_map(resample_data)
        # dataset = dataset.filter(desample_data_0)
        # dataset = dataset.filter(desample_data_0_12_0_16)
        # dataset = dataset.shuffle(buffer_size=1024)
        pass

    return dataset


# 构建训练和测试数据集
train_dataset = build_datasets(train_dataset, is_training=False)
test_dataset = build_datasets(test_dataset, is_training=False)

BATCH_SIZE = 64
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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

model = tf.keras.Sequential([
    tf.keras.layers.Cropping2D(cropping=((30, 25), (0, 0))),
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

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 训练时只使用训练集
history_object = model.fit(train_dataset, validation_data=test_dataset, epochs=30)

model.save('model.h5')

# # 带梯度裁剪的优化器
# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=1e-3,
#     clipnorm=1.0  # 新增梯度裁剪
# )
#
# model.compile(
#     loss='mse',
#     optimizer=optimizer,
#     metrics=['mae']
# )
#
#
# # 改进的余弦退火调度（修改点5：添加warmup）
# def cosine_annealing_with_warmup(epoch):
#     initial_lr = 1e-3
#     warmup_epochs = 5
#     total_epochs = 30
#
#     if epoch < warmup_epochs:  # Warmup阶段
#         return initial_lr * (epoch + 1) / warmup_epochs
#     else:  # 余弦退火阶段
#         decay_epochs = total_epochs - warmup_epochs
#         cos_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / decay_epochs))
#         return initial_lr * cos_decay
#
#
# # 训练配置
# history = model.fit(
#     train_dataset,
#     validation_data=test_dataset,
#     epochs=10,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=10,
#             restore_best_weights=True
#         ),
#         tf.keras.callbacks.ModelCheckpoint(
#             'best_model.h5',
#             save_best_only=True,
#             monitor='val_loss'
#         ),
#         tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup)
#     ]
# )

# # 绘制训练集和验证集的损失曲线
# plt.plot(history.history['loss'])          # 训练损失
# plt.plot(history.history['val_loss'])      # 验证损失
# plt.title('Model Mean Squared Error Loss')        # 标题
# plt.ylabel('Mean Squared Error Loss')             # Y轴标签
# plt.xlabel('Epoch')                               # X轴标签
# plt.legend(['Training Set', 'Validation Set'], loc='upper right')  # 图例
# plt.show()
