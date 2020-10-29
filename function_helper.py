"""
    辅助功能函数
"""
import tensorflow as tf


# create 神经网络
# input_tensor:输入张量
# input_dim：输入神经元节点个数
# output_dim：输出神经元节点个数
# layer_name：当前层名字
# act:激活函数
def nn_layer(input_tensor, input_dim, output_dim, layer_name, l2_loss, act=tf.nn.relu):
    # 定义大节点的名字
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1), dtype=tf.float32, name='weight')
            tf.summary.histogram('weight', weight)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[output_dim]), name='bias')
            tf.summary.histogram('biases', bias)
        with tf.name_scope('wx_plus_b'):
            pre_activate = tf.nn.bias_add(tf.matmul(input_tensor, weight), bias)
            tf.summary.histogram('pre_activate', pre_activate)
        l2_loss += tf.nn.l2_loss(weight)
        l2_loss += tf.nn.l2_loss(bias)
        activation = act(pre_activate, name='activation')
        return activation, l2_loss

