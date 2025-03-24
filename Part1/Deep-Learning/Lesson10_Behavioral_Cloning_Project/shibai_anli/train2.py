
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
CSV_PATH = "data/driving_log.csv"


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
    """数据增强：水平翻转"""
    flipped_image = tf.image.flip_left_right(image)
    flipped_steering = -steering
    return tf.data.Dataset.from_tensors((image, steering)).concatenate(
        tf.data.Dataset.from_tensors((flipped_image, flipped_steering)))


def resample_data(image, steering):
    """根据转向角度重采样数据"""
    abs_steer = tf.abs(steering)

    repeat_times = tf.case([
        (tf.greater_equal(abs_steer,0.4) , lambda :tf.constant(2,tf.int64)),
        (tf.greater_equal(abs_steer,0.1),lambda :tf.constant(1,tf.int64))
    ],default = lambda :tf.constant(1,tf.int64))

    return tf.data.Dataset.from_tensors((image, steering)).repeat(repeat_times)

def desample_data(image,steering):

    should_downsample = tf.logical_and(
        tf.greater_equal(steering,0.0),
        tf.less(steering,0.04)
    )

    keep = tf.cond(
        should_downsample,
        lambda :tf.random.uniform(shape=[],seed=42)< 0.015,
        lambda :tf.constant(True)
    )

    return keep


def build_datasets(line_datasets, is_training=True):
    """构建数据管道"""
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
        dataset = dataset.flat_map(augment_data)
        dataset = dataset.flat_map(resample_data)
        dataset = dataset.filter(desample_data)
        #dataset = dataset.cache()
        #dataset = dataset.shuffle(buffer_size=1024)

    return dataset


# 读取数据集
full_dataset = tf.data.TextLineDataset(CSV_PATH).skip(1)
count = full_dataset.reduce(0, lambda x, _: x + 1).numpy()
full_dataset = full_dataset.shuffle(1024)
total_train = int(count * 0.8)

train_dataset = full_dataset.take(total_train)
test_dataset = full_dataset.skip(total_train)

# 构建训练和测试数据集
train_dataset = build_datasets(train_dataset, is_training=True)
test_dataset = build_datasets(test_dataset, is_training=False)

BATCH_SIZE = 64
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)




# 收集转向角数据
steering_angles = []
for sample in train_dataset.unbatch():
    steering_angles.append(sample[1].numpy())
steering_angles = np.array(steering_angles)

# 使用numpy计算直方图数据（保持与原始数据相同的区间设置）
hist_counts, bin_edges = np.histogram(
    steering_angles,
    bins=50,
    range=(-1.0, 1.0)  # 显式设置范围确保一致性
)

# 打印数据分布
print("直方图区间分布 [左边界, 右边界) : 样本数量 占比")
print(f"{'转向区间':<25} | {'样本数量':<8} | {'占比 (%)':<6}")
print("-" * 45)

total_samples = len(steering_angles)
for i in range(len(hist_counts)):
    left = bin_edges[i]
    right = bin_edges[i+1]
    count = hist_counts[i]
    percent = (count / total_samples) * 100

    # 添加关键区间标注
    range_str = f"[{left:.4f}, {right:.4f})"
    if abs(left) < 0.1:
        range_str += " (直行)"
    elif left > 0.3:
        range_str += " (急右转)"
    elif right < -0.3:
        range_str += " (急左转)"

    print(f"{range_str:<25} | {count:<8} | {percent:.2f}%")

# 扩展统计摘要
print("\n关键统计：")
print(f"总样本数: {total_samples}")
print(f"中位数: {np.median(steering_angles):.4f}")
print(f"均值: {np.mean(steering_angles):.4f}")
print(f"标准差: {np.std(steering_angles):.4f}")
print(f"最小转向角: {np.min(steering_angles):.4f}")
print(f"最大转向角: {np.max(steering_angles):.4f}")
print(f"|θ| < 0.1 样本占比: {np.mean(np.abs(steering_angles) < 0.1 ) * 100:.2f}%")
print(f"|θ| > 0.4 样本占比: {np.mean(np.abs(steering_angles) > 0.4) * 100:.2f}%")





def build_model(input_shape=(160, 320, 3)):
    """构建仅使用图像输入的模型"""
    image_input = layers.Input(shape=input_shape, name='image_input')

    # 预处理层
    x = layers.Cropping2D(cropping=((60, 25), (0, 0)))(image_input)
    x = layers.Resizing(66, 200, interpolation='bilinear')(x)
    x = layers.LayerNormalization()(x)

    # 卷积模块
    x = layers.Conv2D(24, 5, strides=2, activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(36, 5, strides=2, activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(48, 5, strides=2, activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', kernel_regularizer='l2')(x)

    # 全连接层
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(50, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dense(10, activation='relu', kernel_regularizer='l2')(x)
    output = layers.Dense(1)(x)

    return Model(inputs=image_input, outputs=output)


# 创建并编译模型
model = build_model()
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae']
)


def cosine_annealing(epoch):
    initial_lr = 1e-3
    total_epochs = 30

    # 余弦退火公式
    cos_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    return initial_lr * cos_decay


# 训练配置
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=30,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_loss'
        ),
        # 添加余弦退火回调并移除原学习率回调
        tf.keras.callbacks.LearningRateScheduler(cosine_annealing)
    ]
)