from matplotlib import pyplot as plt
# 报错经历：
# tf.stack()是转换的python列表，而非一个个的元素
# tf.data.TextLineDataset创建的dataset读取到csv的数据时，每个单元格以都逗号隔开。
# tf.



from tensorflow.keras import layers,Model
import tensorflow as tf
import numpy as np
import os

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
    """读取每一行CSV，返回图像路径，当前速度，转向角度"""

    line = tf.strings.split(line, ",")
    steering = tf.strings.to_number(line[3], out_type=tf.float32)
    speed = tf.strings.to_number(line[6], out_type=tf.float32)

    paths = tf.stack([
        tf.strings.strip(line[0]),
        tf.strings.strip(line[1]),
        tf.strings.strip(line[2])
    ])
    steers = tf.stack([steering, steering + 0.2, steering - 0.2])
    speeds = tf.stack([speed, speed, speed])

    return paths, speeds, steers


def read_process_image(image_path):
    """ 读取图片，并利用类型转换，进行归一化处理 """

    file = tf.io.read_file(IMAGE_PATH + tf.strings.split(image_path, "/")[-1])
    image_file = tf.io.decode_jpeg(file, channels=3)
    image_file = tf.image.resize(image_file, [160, 320])
    return tf.image.convert_image_dtype(image_file, dtype=tf.float32) - 0.5


def is_steering(image_speed, steering):
    return tf.abs(steering) > 0.05


def is_straight(image_speed, steering):
    return tf.abs(steering) <= 0.05


def augment_data(image_speed, steering):

    image,speed = image_speed
    # speed = tf.reshape(speed,[])
    # steering =tf.reshape(steering,[])

    flipped_image = tf.image.flip_left_right(image)
    flipped_speed = speed
    flipped_steering = -steering

    return tf.data.Dataset.from_tensors(((image, speed), steering)).concatenate(
        tf.data.Dataset.from_tensors(((flipped_image, flipped_speed), flipped_steering)))


def resample_data(img_speed, steering):
    image, speed = img_speed

    repeat_times = tf.cond(
        tf.abs(steering) > 0.3,
        lambda: tf.constant(9, tf.int64),
        lambda: tf.cond(
            tf.abs(steering) >= 0.05,
            lambda: tf.constant(2, tf.int64),
            lambda: tf.constant(1, tf.int64)
        )
    )

    return tf.data.Dataset.from_tensors(((image, speed), steering)).repeat(repeat_times)


def build_datasets(line_datasets, is_training=True):
    """ 构建数据管道 """
    dataset = line_datasets.cache()

    # 读取路径，速度，标签,
    # dataset中的每一个元素结构变为（paths,speeds,steers），其中每个tensor的shape（3,1）
    dataset = dataset.map(
        lambda line: parse_line(line),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 展开数据
    # dataset中每一个元素结构变为（path,speed,steering）,其中每个tensor的shape为（1，）
    dataset = dataset.interleave(
        lambda path, speed, steering: tf.data.Dataset.from_tensor_slices((path, speed, steering)),
        num_parallel_calls=tf.data.AUTOTUNE,
        block_length=1
    )

    # 载入图片
    dataset = dataset.map(
        lambda path, speed, steering: (
            (read_process_image(path),speed),
            steering)
    )

    # 数据增强
    if is_training:
        # 分离直行数据和转向数据
        straight_dataset = dataset.filter(
            lambda image_speed, steering: is_straight(image_speed, steering)
        )
        steering_dataset = dataset.filter(
            lambda image_speed, steering: is_steering(image_speed, steering)
        )

        # 增强并重采样转向数据
        steering_dataset = steering_dataset.flat_map(augment_data)
        steering_dataset = steering_dataset.flat_map(resample_data)

        return (straight_dataset.concatenate(steering_dataset))

    return dataset

# 读取CSV文件
full_datasets = tf.data.TextLineDataset(CSV_PATH).skip(1)
count = full_datasets.reduce(0, lambda x, _: x + 1).numpy()
full_datasets = full_datasets.shuffle(1024)
total_train = count * 0.8
train_dataset = full_datasets.take(total_train)
test_dataset = full_datasets.skip(total_train)

train_dataset = build_datasets(line_datasets=train_dataset, is_training=True)
test_dataset = build_datasets(line_datasets=test_dataset, is_training=False)

BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)




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
#
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