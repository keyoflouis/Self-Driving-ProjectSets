
import csv
import cv2
import numpy as np
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

Image_path = "data/IMG"
Csv_path = "data/driving_log.csv"

lines = []
images =[]
measurements = []

with open(Csv_path) as csv_file:
    reader = csv.reader(csv_file)
    for i,line in enumerate(reader):
        if i==0:
            continue
        lines.append(line)
for line in lines:
    source_path =line[0]
    filename = source_path.split('/')[-1]

    # 读取图片
    current_path = Image_path +'/' + filename
    image = cv2.imread(current_path)
    images.append(image)

    # 读取角速度
    measurement = float(line[3])
    measurements.append(measurement)


X_train = np.array(images)
X_train = X_train/255.0
Y_train = np.array(measurements)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(160,320,3)),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,Y_train,validation_split=0.2,shuffle=True,
          epochs=10,
          batch_size=64)

model.save('model.h5')