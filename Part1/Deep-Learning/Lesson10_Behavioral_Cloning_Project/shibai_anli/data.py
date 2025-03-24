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
    steers = tf.stack([steering, steering + 0.25, steering - 0.25])
    steers = tf.clip_by_value(steers, -1.0, 1.0)

    return paths, steers


def read_process_image(image_path):
    """读取图片并进行预处理"""
    file = tf.io.read_file(IMAGE_PATH + tf.strings.split(image_path, "/")[-1])
    image_file = tf.io.decode_jpeg(file, channels=3)
    image_file = tf.image.resize(image_file, [160, 320])

    return tf.image.convert_image_dtype(image_file, dtype=tf.float32) - 0.5


def random_flip(image, steering):
    ds = tf.data.Dataset.from_tensors((image,steering))
    ds=tf.cond(
        tf.random.uniform([]) < 1,
        lambda: ds.concatenate(tf.data.Dataset.from_tensors((tf.image.flip_left_right(image), -steering))),
        lambda: ds
    )

    return ds


def random_vertical_crop_batch(image, steering):
    height = tf.shape(image)[0]

    top_radio = tf.random.uniform(shape=[], minval=0.325, maxval=0.425)
    bottom_radio = tf.random.uniform(shape=[], minval=0.075, maxval=0.175)

    top = tf.cast(top_radio * tf.cast(height, tf.float32), tf.int32)
    bottom = tf.cast(bottom_radio * tf.cast(height, tf.float32), tf.int32)

    cropped_image = image[top:-bottom, :, :]
    cropped_image = tf.ensure_shape(cropped_image,[None,320,3])
    cropped_image =tf.image.resize(cropped_image,[32, 128])

    return cropped_image, steering


def random_shadow(image, steering):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    # 生成所有x坐标的序列并打乱顺序
    x_values = tf.range(w)
    shuffled_x = tf.random.shuffle(x_values)
    x1, x2 = shuffled_x[0], shuffled_x[1]

    # 确保x1 < x2
    x1, x2 = tf.minimum(x1, x2), tf.maximum(x1, x2)

    # 创建阴影蒙版
    y = tf.range(h, dtype=tf.float32)
    k = tf.cast(h, tf.float32) / tf.cast(x2 - x1, tf.float32)
    b = -k * tf.cast(x1, tf.float32)
    c = (y - b) / k
    c = tf.cast(c, tf.int32)
    c = tf.clip_by_value(c, 0, w - 1)

    # 生成蒙版并应用阴影
    mask = tf.cast(tf.expand_dims(tf.range(w) < c[:, tf.newaxis], axis=-1), tf.float32)
    shadowed_image = image * (1 - 0.5 * mask)

    return shadowed_image, steering


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
        # 随机翻转
        dataset = dataset.flat_map(random_flip)
        # 随机垂直平移
        dataset = dataset.map(random_vertical_crop_batch, num_parallel_calls=tf.data.AUTOTUNE)
        # 随机阴影
        dataset = dataset.map(random_shadow, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda image,steering:(tf.image.resize(image,[32,128]),steering))

    return dataset


def save_to_tfrecord(dataset, filename):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    with tf.io.TFRecordWriter(filename) as writer:
        for image, steering in dataset:
            image_uint8 = tf.cast((image + 0.5) * 255, tf.uint8)
            image_jpeg = tf.io.encode_jpeg(image_uint8)

            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(image_jpeg.numpy()),
                'steering': _float_feature(steering.numpy())
            }))
            writer.write(example.SerializeToString())


def load_tfrecord(filename):
    """从TFRecord文件加载数据集"""

    def parse_example(example_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'steering': tf.io.FixedLenFeature([], tf.float32)
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_jpeg(parsed['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32) - 0.5  # 恢复预处理
        return image, parsed['steering']

    return tf.data.TFRecordDataset(filename).map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)


IMAGE_PATH = "../data/IMG/"
CSV_PATH = "../data/driving_log_balanced.csv"

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
train_dataset_row = train_dataset
test_dataset_row = test_dataset

save_to_tfrecord(train_dataset_row, "train_preprocessed.tfrecords")
save_to_tfrecord(test_dataset_row, "test_preprocessed.tfrecords")

train_dataset = load_tfrecord("train_preprocessed.tfrecords")
test_dataset = load_tfrecord("test_preprocessed.tfrecords")


# 绘制转向角分布直方图
steering_angles = []
for sample in train_dataset:
    steering_angles.append(sample[1].numpy())
plt.hist(steering_angles, bins=50)
plt.title("Steering Angle Distribution")
plt.show()

# 收集转向角数据
steering_angles = []
for sample in train_dataset:
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

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)