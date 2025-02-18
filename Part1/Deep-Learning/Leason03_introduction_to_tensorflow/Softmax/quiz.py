## Solution is available in the other "solution.py" tab
#import tensorflow as tf
#
#
#def run():
#    output = None
#    logit_data = [2.0, 1.0, 0.1]
#    logits = tf.placeholder(tf.float32)
#
#    # TODO: Calculate the softmax of the logits
#    softmax = tf.nn.softmax(logits)
#
#    with tf.Session() as sess:
#        # TODO: Feed in the logit data
#        output = sess.run(softmax, feed_dict={logits: logit_data})
#
#    return output
#

import tensorflow as tf

def run():
    logit_data = [2.0, 1.0, 0.1]
    
    # 将 logit_data 转换为 TensorFlow 张量
    logits = tf.convert_to_tensor(logit_data, dtype=tf.float32)
    
    # 计算 softmax
    softmax = tf.nn.softmax(logits)
    
    # 直接输出结果
    output = softmax.numpy()
    
    return output

# 调用函数并打印结果
print(run())