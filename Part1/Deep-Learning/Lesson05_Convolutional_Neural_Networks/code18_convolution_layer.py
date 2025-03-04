import tensorflow as tf

image_height = 10
image_width = 10
color_channnels = 3

k_output = 64
filte_size = (5, 5)
strides = (2, 2)

# 创建层，来隐式创建计算图
inputs = tf.keras.Input(shape=(image_height, image_width, color_channnels))

# 创建层，来隐式创建计算图
x = tf.keras.layers.Conv2D(
    filters=k_output,
    kernel_size=filte_size,
    strides=strides,

    padding='same',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.truncated_normal(),
    bias_initializer='zeros',
    activation='relu'
)(inputs)

# 创建模型
model = tf.keras.Model(inputs=inputs, outputs=x)

# 8个样本，10*10*3的shape
dummy_data = tf.random.normal([8, 10, 10, 3])

# 输出8个样本,5*5*64的shape
print(model(dummy_data).shape)
