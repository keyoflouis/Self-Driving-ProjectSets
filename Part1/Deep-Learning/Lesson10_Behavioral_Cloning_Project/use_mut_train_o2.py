import tensorflow as tf
import os

from keras.dtensor.layout_map import layout_map_scope
from tensorflow.keras import layers,Model
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
    speed = tf.strings.to_number(tf.strings.strip(part[6]), out_type=tf.float32)

    speeds = tf.stack([
        speed,
        speed,
        speed
    ])

    steerings = tf.stack([
        steering_center,
        steering_center + correction,
        steering_center - correction
    ])

    return image_path,speeds,steerings


def process_image(path):
    filename = IMG_PATH + tf.strings.split(path, '/')[-1]

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32) - 0.5
    return image


def augment_data(image, speed,steering):
    # 确保steering是标量
    steering = tf.reshape(steering, [])
    speed = tf.reshape(speed,[])

    # 数据增强
    flipped_image = tf.image.flip_left_right(image)
    flipped_steering = -steering
    flipped_speed = speed

    # 创建包含两个样本的数据集
    return tf.data.Dataset.from_tensors(
        ((image, speed),steering)
    ).concatenate(
        tf.data.Dataset.from_tensors(((flipped_image, flipped_speed),flipped_steering))
    )


def build_dataset(line_dataset, batch_size=32, is_training=True, correction=0.2):
    ds = line_dataset.cache()

    # 并行读取CSV
    ds = ds.map(
        lambda line: parse_line(line, correction),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 并行处理图像
    ds = ds.interleave(
        lambda paths,speeds,sts: tf.data.Dataset.from_tensor_slices((paths,speeds,sts)),
        num_parallel_calls=tf.data.AUTOTUNE,
        block_length=1
    )

    # 加载图片
    ds = ds.map(
        lambda path,speed,st: (process_image(path),speed, st),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 数据增强，水平翻转
    ds = ds.flat_map(augment_data)

    if is_training:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


lines = tf.data.TextLineDataset(CSV_PATH).skip(1)
lines = lines.shuffle(10000, seed=42).cache()
total_samples = len(list(lines.as_numpy_iterator()))
train_samples = int(0.8 * total_samples)
test_samples = total_samples - train_samples
train_lines = lines.take(train_samples)
test_lines = lines.skip(train_samples)

# 数据集构建
train_dataset = build_dataset(train_lines, batch_size=32, is_training=True)
test_dataset = build_dataset(test_lines, batch_size=32, is_training=False)

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
    speed_features = layers.Dense(16,activation='relu')(speed_input)
    speed_features = layers.Dense(16,activation='relu')(speed_features)

    merged = layers.concatenate([image_features,speed_features])

    x = layers.Dense(100,activation='relu',kernel_regularizer='l2')(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(50,activation='relu',kernel_regularizer='l2')(x)
    x = layers.Dense(10,activation='relu',kernel_regularizer='l2')(x)
    output =layers.Dense(1)(x)

    return Model(inputs=[image_input, speed_input], outputs=output)

model = build_model()


model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['mae']
)

# 修改后的训练配置
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=30,  # 设置为30个epoch
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # 增加耐心到5个epoch
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2  # 添加学习率自适应
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_loss'
        )
    ]
)

# 绘制训练集和验证集的损失曲线
plt.plot(history.history['loss'])          # 训练损失
plt.plot(history.history['val_loss'])      # 验证损失
plt.title('Model Mean Squared Error Loss')        # 标题
plt.ylabel('Mean Squared Error Loss')             # Y轴标签
plt.xlabel('Epoch')                               # X轴标签
plt.legend(['Training Set', 'Validation Set'], loc='upper right')  # 图例
plt.show()