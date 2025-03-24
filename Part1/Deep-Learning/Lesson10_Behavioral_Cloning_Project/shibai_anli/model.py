import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from data import train_dataset, test_dataset

# 数据预处理配置
BATCH_SIZE = 64  # 根据显存调整
IMG_SHAPE = (32, 128, 3)
AUTOTUNE = tf.data.experimental.AUTOTUNE

# 训练配置
EPOCHS = 100
LEARNING_RATE = 1e-4

# 构建模型
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, activation='relu', input_shape=IMG_SHAPE),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(100, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(20, activation='relu'),
    layers.Dense(1)
])

# 配置优化器
optimizer = Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

# 编译模型
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mse', 'mae']
)

# 配置回调函数
callbacks = [
    callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        save_best_only=True,
        monitor='val_loss'
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )
]

# 训练模型
history = model.fit(
    train_dataset.prefetch(AUTOTUNE),
    epochs=EPOCHS,
    validation_data=test_dataset.prefetch(AUTOTUNE),
    callbacks=callbacks,
    verbose=1
)

# 保存最终模型
model.save("final_model.h5")