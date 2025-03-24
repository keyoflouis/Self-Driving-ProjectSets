import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers, Model, utils
from matplotlib import pyplot as plt

# GPU配置
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 替代原来的策略配置
from tensorflow.keras import mixed_precision

# 配置混合精度策略
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)



# 路径配置
IMG_PATH = "data/IMG/"
CSV_PATH = "data/driving_log.csv"


# 自定义注意力模块
class CBAM(layers.Layer):
    def __init__(self, ratio=8, kernel_size=7,  ** kwargs):  # 添加**kwargs
        super().__init__(**kwargs)  # 传递kwargs给父类
        self.ratio = ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channels = input_shape[-1]

        # 通道注意力
        self.channel_avg = layers.GlobalAveragePooling2D()
        self.channel_max = layers.GlobalMaxPooling2D()
        self.channel_fc1 = layers.Dense(channels // self.ratio, activation='relu')
        self.channel_fc2 = layers.Dense(channels, activation='sigmoid')

        # 空间注意力
        self.spatial_conv = layers.Conv2D(1, self.kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        # 通道注意力
        avg_out = self.channel_fc2(self.channel_fc1(self.channel_avg(inputs)))
        max_out = self.channel_fc2(self.channel_fc1(self.channel_max(inputs)))
        channel = layers.add([avg_out, max_out])
        channel = tf.sigmoid(channel)
        channel_out = inputs * tf.reshape(channel, [-1, 1, 1, channel.shape[-1]])

        # 空间注意力
        spatial = self.spatial_conv(channel_out)
        spatial_out = channel_out * spatial

        return spatial_out
    # 新增配置方法
    def get_config(self):
        config = super().get_config()
        config.update({
            "ratio": self.ratio,
            "kernel_size": self.kernel_size
        })
        return config
    # 可选：兼容旧版本
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# 修改后的process_line
def process_line(line):
    parts = tf.strings.split(line, ',')

    # 解析原始数据
    paths = [tf.strings.strip(parts[i]) for i in [0, 1, 2]]  # 三个摄像头路径
    steering = tf.strings.to_number(parts[3], tf.float32)
    speed = tf.strings.to_number(parts[6], tf.float32)

    # 生成多摄像头样本
    return {
        "path": paths,
        "speed": [speed] * 3,
        "steering": [
            steering,
            steering + 0.2,  # 左摄像头校正
            steering - 0.2  # 右摄像头校正
        ],
        "is_curve": tf.cast(tf.abs(steering) > 0.5, tf.int32)
    }

# 弯道增强
def curve_augmentation(image, steering):
    # 基础增强
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    # 弯道专用增强
    if tf.abs(steering) > 0.5:
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_jpeg_quality(image, 70, 100)

    return image


# 修正后的数据管道
def build_dataset(lines, batch_size=32, curve_repeat=3):
    # 展开多摄像头样本
    ds = lines.flat_map(
        lambda line: tf.data.Dataset.from_tensors(process_line(line)).flat_map(
            lambda data: tf.data.Dataset.zip({
                "path": tf.data.Dataset.from_tensor_slices(data["path"]),
                "speed": tf.data.Dataset.from_tensor_slices(data["speed"]),
                "steering": tf.data.Dataset.from_tensor_slices(data["steering"]),
                "is_curve": tf.data.Dataset.from_tensor_slices([data["is_curve"]] * 3)
            })
        )
    )

    # 弯道样本重采样
    def is_curve(sample):
        return sample["is_curve"] == 1

    curve_ds = ds.filter(is_curve).repeat(curve_repeat)
    straight_ds = ds.filter(lambda x: not is_curve(x))
    ds = curve_ds.concatenate(straight_ds)

    # 图像加载与增强
    def load_and_augment(sample):
        image = tf.image.decode_jpeg(tf.io.read_file(IMG_PATH + tf.strings.split(sample["path"], '/')[-1]))
        image = tf.image.convert_image_dtype(image, tf.float32)

        # 弯道增强
        if sample["is_curve"] == 1:
            image = tf.image.random_hue(image, 0.08)
            image = tf.image.random_jpeg_quality(image, 70, 100)

        image = tf.image.convert_image_dtype(image, tf.float32)  # 保持float32输入
        return (image, sample["speed"]), tf.cast(sample["steering"], tf.float32)  # 确保标签类型

    return ds.map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(1024) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)


# 改进的模型架构
def build_model(input_shape=(160, 320, 3)):
    # 图像输入分支
    image_input = layers.Input(shape=input_shape, name='image_input')
    x = layers.Cropping2D(((60, 25), (0, 0)))(image_input)
    x = layers.Resizing(66, 200, interpolation='bilinear')(x)
    x = layers.LayerNormalization()(x)

    # 卷积模块 + CBAM
    x = layers.Conv2D(24, 5, strides=2, activation='relu')(x)
    x = CBAM()(x)
    x = layers.Conv2D(36, 5, strides=2, activation='relu')(x)
    x = CBAM()(x)
    x = layers.Conv2D(48, 5, strides=2, activation='relu')(x)
    x = CBAM()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = CBAM()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    image_features = layers.Flatten()(x)

    # 速度输入分支
    speed_input = layers.Input(shape=(1,), name='speed_input')
    speed_features = layers.Dense(16, activation='relu')(speed_input)
    speed_features = layers.Dense(16, activation='relu')(speed_features)

    # 门控融合
    gate = layers.Dense(image_features.shape[-1], activation='sigmoid')(speed_features)
    gated_features = layers.Multiply()([image_features, gate])
    merged = layers.concatenate([gated_features, speed_features])

    # 回归头
    x = layers.Dense(100, activation='relu')(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(10, activation='relu')(x)

    output = layers.Dense(1, dtype='float32')(x)  # 强制输出层为float32
    return Model(inputs=[image_input, speed_input], outputs=output)


# 修正后的损失函数
class CurveSensitiveLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=3.0):
        super().__init__()
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # 显式转换数据类型
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        mse = tf.square(y_pred - y_true)
        curve_mask = tf.cast(tf.abs(y_true) > 0.5, tf.float32)
        weights = 1.0 + (self.alpha - 1.0) * curve_mask
        return tf.reduce_mean(weights * mse)

# 初始化模型

# 创建优化器时需要显式包装
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)  # 添加损失缩放

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=CurveSensitiveLoss(alpha=3.0),
    metrics=['mae']
)



history = model.fit(
    build_dataset(tf.data.TextLineDataset(CSV_PATH).skip(1).shuffle(10000)),
    epochs=30,
    validation_data=build_dataset(tf.data.TextLineDataset(CSV_PATH).skip(1), curve_repeat=1),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint('best_model1.h5', save_best_only=True)
    ]
)
# 保存最终模型
model.save('final_model.h5')

# 可视化训练过程
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Curve-Optimized Training Progress')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()