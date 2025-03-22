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

Image_path = "data/IMG/"
Csv_path = "data/driving_log.csv"


def load_and_preprocess(line):
    # 载入
    parts = tf.strings.split(line, ",")
    image_path = tf.strings.strip(parts[0])
    row_measurement = tf.strings.strip(parts[3])
    measurement = tf.strings.to_number(row_measurement, out_type=tf.float32)

    # 图像预处理
    image = tf.io.read_file(Image_path + tf.strings.split(image_path, '/')[-1])
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float16)  # 这里会自动转换为[0-1]
    image = (image - 0.5)

    return image, measurement


def load_and_flip(line):
    parts = tf.strings.split(line, ",")
    image_path = tf.strings.strip(parts[0])
    row_measurement = tf.strings.strip(parts[3])
    measurement = tf.strings.to_number(row_measurement, out_type=tf.float32)

    # 图像预处理
    image = tf.io.read_file(Image_path + tf.strings.split(image_path, '/')[-1])
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float16)  # 这里会自动转换为[0-1]
    image = (image - 0.5)

    image = tf.image.flip_left_right(image)
    label = -1 * measurement
    return image, label


# 以TextLineDataset的方式载入CSV文件
lines = tf.data.TextLineDataset(Csv_path).skip(1).cache()
n = lines.reduce(0, lambda x, _: x + 1).numpy()

# 对Lines中的每一行为单位使用map
dataset = lines.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

fliped_datasets = lines.map(load_and_flip, num_parallel_calls=tf.data.AUTOTUNE)

# 合并数据集（关键修改点）
combined_dataset = dataset.concatenate(fliped_datasets)

# 划分训练集和验证集
total_samples = 2 * n
train_size = int(0.8 * total_samples)
# 打乱并划分数据集（关键修改点）
combined_dataset = combined_dataset.shuffle(buffer_size=total_samples)

train_dataset = combined_dataset.take(train_size).shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = combined_dataset.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)




# 模型
model = tf.keras.Sequential([
    #tf.keras.layers.Cropping2D(input_shape=(160,320,3),cropping=((70,25),(0,0))),
    tf.keras.layers.Conv2D(filters=6, kernel_size=(8, 8), input_shape=(160,320,3),padding="SAME", strides=4,
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
model.fit(train_dataset, validation_data=val_dataset, epochs=5)
model.save('model.h5')
