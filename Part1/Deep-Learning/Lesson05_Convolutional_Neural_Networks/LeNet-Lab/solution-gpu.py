# GPU:3060

from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf

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

# 加载处理数据（保持不变）
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

X_val = X_train[55000:]
Y_val = Y_train[55000:]
X_train = X_train[:55000]
Y_train = Y_train[:55000]

# 添加padding
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_val = np.pad(X_val, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# 断言检查
assert len(X_train) == len(Y_train)
assert len(X_val) == len(Y_val)
assert len(X_test) == len(Y_test)

print("\n图像形状:", X_train[0].shape)
print("训练集:   {} 样本".format(len(X_train)))
print("验证集: {} 样本".format(len(X_val)))
print("测试集:       {} 样本\n".format(len(X_test)))


# GPU监控回调类
class GPUMonitor(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
        print(f"\nEpoch {epoch + 1} 开始 - GPU内存使用: {gpu_stats['current'] / 1024 ** 2:.1f} MB")

    def on_epoch_end(self, epoch, logs=None):
        gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
        print(f"\n Epoch {epoch + 1} 结束 - GPU内存峰值: {gpu_stats['peak'] / 1024 ** 2:.1f} MB")


# 修改后的LeNet模型（适配混合精度）
def LeNet():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), activation="relu",
                               input_shape=(32, 32, 1), dtype='float32'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(16, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        # 输出层保持float32精度
        tf.keras.layers.Dense(10, activation='linear', dtype='float32')
    ])
    return model


# 编译模型
model = LeNet()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练参数（可动态调整）
initial_batch_size = 512  # 初始批次大小（根据GPU显存调整）
epochs = 10

try:
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=initial_batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[GPUMonitor()],  # 添加GPU监控
        verbose=1
    )
except tf.errors.ResourceExhaustedError:
    print("\n显存不足！尝试减小批次大小...")
    adjusted_batch_size = initial_batch_size // 2
    print(f"使用新批次大小: {adjusted_batch_size}")
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=adjusted_batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[GPUMonitor()],
        verbose=1
    )

# 保存模型
model.save('lenet_model')

# 评估测试集
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print("\n最终测试精度 = {:.3f}".format(test_acc))