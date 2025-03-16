

import csv
import cv2
import numpy as np
import tensorflow as tf
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# GPU配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# 混合精度配置
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print("Compute dtype:", policy.compute_dtype)
print("Variable dtype:", policy.variable_dtype)

Image_path = "data/IMG"
Csv_path = "data/driving_log.csv"

lines = []
images = []
measurements = []

# 数据加载（保持原始像素值）
with open(Csv_path) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        if reader.line_num == 1: continue  # 跳过标题行
        lines.append(line)

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]

    # 读取并转换颜色空间
    current_path = os.path.join(Image_path, filename)
    image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
    images.append(image)

    # 读取转向角度
    measurements.append(float(line[3]))

X_train = np.array(images)  # 保持uint8格式 [0, 255]
Y_train = np.array(measurements)

# NVIDIA端到端网络架构
model = tf.keras.Sequential([
    # 输入预处理层
    tf.keras.layers.InputLayer(input_shape=(160, 320, 3)),
    tf.keras.layers.Cropping2D(cropping=((70, 25), (0, 0))),  # 裁剪为65x320
    tf.keras.layers.Resizing(66, 200),  # NVIDIA标准输入尺寸
    tf.keras.layers.Lambda(lambda x: x / 255.0 - 0.5),  # 归一化

    # 卷积特征提取层
    tf.keras.layers.Conv2D(24, 5, strides=2, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(36, 5, strides=2, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(48, 5, strides=2, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', activation='relu'),

    # 全连接决策层
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, dtype='float32')  # 确保最终输出精度
])

# 混合精度优化配置
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

model.compile(loss='mse', optimizer=optimizer)

# 训练配置
history = model.fit(
    X_train, Y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)

model.save('nvidia_model.h5')

# 打印模型结构
model.summary()