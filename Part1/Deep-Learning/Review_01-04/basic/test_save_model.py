import tensorflow as tf

# 定义模型
model = tf.keras.Sequential([...])
model.compile(optimizer='adam', loss='mse')

# 创建回调：每个 epoch 保存一次权重
checkpoint_path = "checkpoints/model.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # 仅保存权重（不保存模型结构）
    save_freq='epoch'        # 每个 epoch 保存一次
)

# 训练模型
model.fit(train_data, epochs=10, callbacks=[checkpoint_callback])

# 恢复权重（需要先构建相同结构的模型）
new_model = tf.keras.Sequential([...])  # 必须与原模型结构相同
new_model.compile(optimizer='adam', loss='mse')
new_model.load_weights(checkpoint_path)