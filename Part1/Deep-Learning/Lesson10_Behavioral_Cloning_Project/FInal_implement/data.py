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
        # dataset = dataset.flat_map(augment_data)
        # dataset = dataset.flat_map(resample_data)
        # dataset = dataset.shuffle(buffer_size=1024)
        pass

    return dataset

IMAGE_PATH = "data/IMG/"
CSV_PATH = "data/driving_log_balanced.csv"

# 读取数据集
full_dataset = tf.data.TextLineDataset(CSV_PATH).skip(1)
count = full_dataset.reduce(0, lambda x, _: x + 1).numpy()
full_dataset = full_dataset.shuffle(1024)
total_train = int(count * 0.8)

train_dataset = full_dataset.take(total_train)
test_dataset = full_dataset.skip(total_train)

# 构建训练和测试数据集
train_dataset = build_datasets(train_dataset, is_training=False)
test_dataset = build_datasets(test_dataset, is_training=False)

BATCH_SIZE = 64
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
