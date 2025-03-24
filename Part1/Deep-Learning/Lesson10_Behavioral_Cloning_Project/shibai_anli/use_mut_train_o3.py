import tensorflow as tf
import os
import numpy as np
from keras.dtensor.layout_map import layout_map_scope
from tensorflow.keras import layers, Model
from matplotlib import pyplot as plt

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

IMG_PATH = "data/IMG/"
CSV_PATH = "data/driving_log.csv"

def resample_function(img_speed, steering):
    """弯道样本重采样函数（参数结构修正版）"""
    image, speed = img_speed
    abs_steering = tf.abs(steering)

    # 嵌套条件判断
    repeat_times = tf.cond(
        abs_steering > 0.5,
        lambda: tf.constant(4, dtype=tf.int64),  # 转向>0.5时重复4次
        lambda: tf.cond(
            abs_steering > 0.3,
            lambda: tf.constant(2, dtype=tf.int64),  # 0.3<转向<=0.5时重复2次
            lambda: tf.constant(1, dtype=tf.int64)  # 其他情况重复1次
        )
    )

    return tf.data.Dataset.from_tensors(((image, speed), steering)).repeat(repeat_times)

def parse_line(line, correction):
    """解析CSV行数据"""
    part = tf.strings.split(line, ',')
    image_path = [tf.strings.strip(part[i]) for i in [0, 1, 2]]
    steering_center = tf.strings.to_number(tf.strings.strip(part[3]), out_type=tf.float32)
    speed = tf.strings.to_number(tf.strings.strip(part[6]), out_type=tf.float32)

    speeds = tf.stack([speed, speed, speed])
    steerings = tf.stack([
        steering_center,
        steering_center + correction,
        steering_center - correction
    ])
    return image_path, speeds, steerings


def process_image(path):
    """图像预处理"""
    filename = IMG_PATH + tf.strings.split(path, '/')[-1]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.image.convert_image_dtype(image, tf.float32) - 0.5


def augment_data(image, speed, steering):
    """数据增强：随机水平翻转"""
    steering = tf.reshape(steering, [])
    speed = tf.reshape(speed, [])

    flipped_image = tf.image.flip_left_right(image)
    return tf.data.Dataset.from_tensors(
        ((image, speed), steering)
    ).concatenate(
        tf.data.Dataset.from_tensors(((flipped_image, speed), -steering))
    )


def build_dataset(line_dataset, batch_size=32, is_training=True, correction=0.2):
    """构建训练数据集管道（最终修正版）"""
    ds = line_dataset.cache()

    # 解析CSV数据
    ds = ds.map(
        lambda line: parse_line(line, correction),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 展开多摄像头数据（返回三元素结构：path, speed, steering）
    ds = ds.interleave(
        lambda paths, speeds, sts: tf.data.Dataset.from_tensor_slices((paths, speeds, sts)),
        num_parallel_calls=tf.data.AUTOTUNE,
        block_length=1
    )

    # 图像预处理（保持三元素结构：image, speed, steering）
    ds = ds.map(
        lambda path, speed, st: (process_image(path), speed, st),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 数据增强（转换为两元素结构：((image, speed), steering)）
    ds = ds.flat_map(lambda img, speed, st: augment_data(img, speed, st))

    # 训练时应用弯道重采样
    if is_training:
        # 将两元素结构直接传给重采样函数
        ds = ds.flat_map(resample_function)
        ds = ds.shuffle(1024)


    # 转换为模型需要的输入格式：((image, speed), steering)
    ds = ds.map(
        lambda img_speed, st: ( (img_speed[0], img_speed[1]), st ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 数据集拆分
lines = tf.data.TextLineDataset(CSV_PATH).skip(1)
lines = lines.shuffle(10000, seed=42).cache()
total_samples = len(list(lines.as_numpy_iterator()))
train_lines = lines.take(int(0.8 * total_samples))
test_lines = lines.skip(int(0.8 * total_samples))

# 构建数据集
train_dataset = build_dataset(train_lines, is_training=True)
test_dataset = build_dataset(test_lines, is_training=False)




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

    image_input =layers.Input(shape=input_shape,name='image_input')
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

    image_features = layers.Flatten()(x)

    # 速度分支
    speed_input =layers.Input(shape=(1,),name='speed_input')
    speed_normalized = layers.LayerNormalization()(speed_input)
    speed_features = layers.Dense(16,activation='relu')(speed_normalized)
    speed_features = layers.Dense(16,activation='relu')(speed_features)

    merged = layers.concatenate([image_features,speed_features])

    x = layers.Dense(100,activation='relu',kernel_regularizer='l2')(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(50,activation='relu',kernel_regularizer='l2')(x)
    x = layers.Dense(10,activation='relu',kernel_regularizer='l2')(x)
    output =layers.Dense(1)(x)

    return Model(inputs=[image_input, speed_input], outputs=output)

# model = build_model()
#
#
# model.compile(
#     loss='mse',
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     metrics=['mae']
# )
#
# # 修改后的训练配置
# history = model.fit(
#     train_dataset,
#     validation_data=test_dataset,
#     epochs=30,  # 设置为30个epoch
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=10,
#             restore_best_weights=True,
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=2
#         ),
#         tf.keras.callbacks.ModelCheckpoint(
#             'best_model.h5',
#             save_best_only=True,
#             monitor='val_loss'
#         )
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