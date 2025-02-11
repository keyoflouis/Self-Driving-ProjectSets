# x = tf.placeholder(tf.string)
# y = tf.placeholder(tf.int32)
# z = tf.placeholder(tf.float32)
#
# with tf.Session() as sess:
#     output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf

x = tf.constant("hello tensorflow")

out = x.numpy()

print(out)


# 基础数学运算

a = tf.constant([1,2,3])
b = tf.constant([2,3,4])

c = tf.add(a,b)
print(c.numpy())

d = tf.subtract(b,a)
print(d.numpy())

e = tf.multiply(a,b)
print(e.numpy())

f = tf.constant([1.5,2.6,4.1])
g = tf.cast(f,tf.int32)

print(g.numpy())


# import tensorflow as tf
#
# x = tf.add(5,2)
# y = tf.subtract(10,4)
# z = tf.multiply(2,5)
# w = tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))
#
# with tf.Session() as sess:
#     output = sess.run(x)
#     print(output)
#     output = sess.run(y)
#     print(output)
#     output = sess.run(z)
#     print(output)
#     output = sess.run(w)
#     print(output)
#
# # TODO: Convert the following to TensorFlow:
# x = 10
# y = 2
# z = x/y - 1
# x = tf.constant(x)
# y = tf.constant(y)
#
# # TODO: Print z from a session
# z = tf.subtract(tf.cast(tf.divide(x,y),tf.int32),tf.constant(1))
# print()
# print("------------Algorithm Solution starts here------------")
# print()
# with tf.Session() as sess:
#     print("x = ",sess.run(x))
#     print("y = ",sess.run(y))
#     print("z = x/y - 1, which evaluates to: ",sess.run(z))


# 启用 eager execution
tf.config.run_functions_eagerly(True)

# 直接进行计算
x = tf.add(5, 2)
y = tf.subtract(10, 4)
z = tf.multiply(2, 5)
w = tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))

# 直接打印结果
print(x.numpy())
print(y.numpy())
print(z.numpy())
print(w.numpy())

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x / y - 1
x = tf.constant(x)
y = tf.constant(y)

# TODO: Print z from a session
z = tf.subtract(tf.cast(tf.divide(x, y), tf.int32), tf.constant(1))

print()
print("------------Algorithm Solution starts here------------")
print()
print("x = ", x.numpy())
print("y = ", y.numpy())
print("z = x/y - 1, which evaluates to: ", z.numpy())