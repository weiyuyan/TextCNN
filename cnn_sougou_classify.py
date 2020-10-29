"""
    功能：CNN文本分类
    目前的模型：wordembedding_layer,cnn_layer,full_connected_layer
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import my_data_helper
from sklearn.model_selection import train_test_split

# 超参
sequence_length = 0
classes_num = 9
vocabulary_size = 0
embedding_size = 100
l2_lambda = 0.001
batch_size = 50     # 这是说一个batch有50个样本数据
epochs = 1000
learning_rate = 1e-3
drop_keep_prob = 0.5
filters_height = [2, 3, 4]
filter_num_per_height = [100, 100, 100]
max_steps = 10000
log_every_n = 1000
save_every_n = 500


train_logdir='D:/my_AI/cnn_sougou_text_classify/graph/train'
test_logdir='D:/my_AI/cnn_sougou_text_classify/graph/test'

print('Loading data...')
train_input, train_label, vocabulary, vocabulary_inv = my_data_helper.load_data()
vocabulary_size = len(vocabulary)
print(train_input)
print(train_label)
print(vocabulary)
print(vocabulary_inv)

# 再对数据打乱处理
shuffle_indices = np.random.permutation(np.arange(len(train_input)))
x_shuffled = train_input[shuffle_indices]
y_shuffled = train_label[shuffle_indices]
# 在此处就要切分好训练集和测试集，先按照8:2的方式
x_shuffled_train, x_shuffled_test, y_shuffled_train, y_shuffled_test = train_test_split(x_shuffled, y_shuffled, test_size=0.2)
s1 = str(np.shape(x_shuffled))
s2 = str(np.shape(x_shuffled_test))
s3 = str(np.shape(x_shuffled_train))
print('总样本个数：' + s1)
print('训练集：'+s2)
print('测试集：'+s3)

# 测试分离出的batch是否可用
# # 获取训练数据,shuffled过后的数据
# a = my_data_helper.get_batch(zip(x_shuffled_train, y_shuffled_train), batch_size, epochs)
# # 获取测试数据
# b = my_data_helper.get_batch(zip(x_shuffled_test, y_shuffled_test), batch_size, epochs)
# for i, j in zip(a, b):
#     # print(i)
#     print(j)



# 确定序列的长度
sequence_length = train_input.shape[1]
print('该训练集中词汇表大小：{:d}'.format(vocabulary_size))
print('一个句子序列的长度为：{:d}'.format(sequence_length))


# 构建数据流图
graph = tf.Graph()
with graph.as_default():
    # 训练数据
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, sequence_length], name='inputs')
    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.float32, [None, classes_num], name='labels')
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('l2_loss'):
        l2_loss = tf.constant(0.0, tf.float32, name='l2_loss')

    # 词向量
    with tf.device('/cpu:0'):
        with tf.name_scope('embedding_layer'):
            # 词嵌入库
            embeddings = tf.Variable(tf.random_normal([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')
            # 输入数据是每个句子的单词的索引id，则就可以直接查表，得到改词的词向量
            embed = tf.nn.embedding_lookup(embeddings, inputs, name='embed')
            # 作为卷积的直接输入。卷积要求必须有通道数，虽然文本的厚度为1，只有一个通道，但要加上
            conv_inputs = tf.expand_dims(embed, -1)

        with tf.name_scope('conv_pooling_layer'):
            # 存储处理好后的特征,注意feature要加s，不要混淆
            features_pooled = []
            for filter_height, filter_num in zip(filters_height, filter_num_per_height):
                with tf.name_scope('conv_filter'):
                    # 卷积核四个维度[高，宽，通道，个数]
                    conv_filter = tf.Variable(tf.truncated_normal([filter_height, embedding_size, 1, filter_num], stddev=0.1), name='conv_filer')
                # 卷积操作
                with tf.name_scope('conv'):
                    conv = tf.nn.conv2d(conv_inputs, conv_filter, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                # 偏置,一个滤波器对应一个偏置
                with tf.name_scope('bias'):
                    bias = tf.Variable(tf.constant(0.1, shape=[filter_num]))
                # 非线性，Relu
                with tf.name_scope('Relu'):
                    feature_map = tf.nn.relu(tf.nn.bias_add(conv, bias), name='Relu')
                # 池化
                # tf.nn.max_pool(value,ksize,strides,padding)
                # value: 4维张量；ksize：包含4个元素的1维张量，对应输入张量每一维度窗口的大小,就是kernel size；
                with tf.name_scope('max_pooling'):
                    feature_pooled = tf.nn.max_pool(feature_map, ksize=[1, sequence_length-filter_height+1, 1, 1],
                                                    strides=[1, 1, 1, 1], padding='VALID', name='max_pooling')
                features_pooled.append(feature_pooled)

        with tf.name_scope('full_connected_layer'):
            filter_num_total = sum(filter_num_per_height)
            # 就是平铺,tf.concat(features_pooled, 3):第4个维度进行拼接
            features_pooled_flat = tf.reshape(tf.concat(features_pooled, 3), [-1, filter_num_total])
            # 该层要dropout
            with tf.name_scope('drop_out'):
                features_pooled_flat_drop = tf.nn.dropout(features_pooled_flat, keep_prob=keep_prob, name='drop_out')
            with tf.name_scope('weight'):
                weight = tf.Variable(tf.truncated_normal(shape=[filter_num_total, classes_num], dtype=tf.float32), name='weight')
                tf.summary.histogram('weight', weight)
            with tf.name_scope('bias'):
                bias = tf.Variable(tf.constant(0.1, shape=[classes_num]), name='bias')
                tf.summary.histogram('bias', bias)
            # L2范数正则化
            with tf.name_scope('L2'):
                l2_loss += tf.nn.l2_loss(weight)
                l2_loss += tf.nn.l2_loss(bias)
            # xw_plus_b
            with tf.name_scope('xw_plus_b'):
                scores = tf.nn.xw_plus_b(features_pooled_flat_drop, weight, bias, name='xw_plus_b')
                tf.summary.histogram('xw_plus_b', scores)
                # 保存每个标签值的得分，以便在预测时候使用。将预测值放入该列表中
                tf.add_to_collection('pred_network', scores)
            # cross_entropy loss
            with tf.name_scope('softmax_cross_entropy'):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=scores, name='losses')
            # loss, is a scalar
            with tf.name_scope('train_loss'):
                train_loss = tf.reduce_mean(losses) + l2_lambda * l2_loss
                tf.summary.scalar('train_loss', train_loss)
            with tf.name_scope('test_loss'):
                test_loss = tf.reduce_mean(losses) + l2_lambda * l2_loss
                tf.summary.scalar('test_loss', test_loss)

            # 预测
            with tf.name_scope('prediction'):
                predictions = tf.argmax(scores, 1)
                correct_predictions = tf.equal(predictions, tf.argmax(labels, 1), name='correct_predictions')
            # accuracy
            with tf.name_scope('train_accuracy'):
                train_accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
                tf.summary.scalar('train_accuracy', train_accuracy)
            # accuracy
            with tf.name_scope('test_accuracy'):
                test_accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
                tf.summary.scalar('test_accuracy', test_accuracy)

# 运行数据流图
with tf.Session(graph=graph) as sess:
    # 全局迭代数
    global_step = tf.Variable(0, trainable=False)
    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # train
    with tf.name_scope('train_op'):
        train_op = optimizer.minimize(train_loss, global_step=global_step)
    sess.run(tf.global_variables_initializer())
    # 获取训练数据,shuffled过后的数据
    batches_train = my_data_helper.get_batch(zip(x_shuffled_train, y_shuffled_train), batch_size, epochs)
    # 获取测试数据
    batches_test = my_data_helper.get_batch(zip(x_shuffled_test, y_shuffled_test), batch_size, epochs)

    # 所有汇总
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir=train_logdir, graph=graph)
    writer1 = tf.summary.FileWriter(logdir=test_logdir)
    # saver
    saver = tf.train.Saver()
    temp = 1

    # 迭代训练
    for batch_train, batch_test in zip(batches_train, batches_test):

        temp += 1
        if temp % 4 == 0:
            # 获取测试数据，如果训练数据是测试数据的4倍，那么每4步显示一次测试的loss和acc
            x_batch_test, y_batch_test = zip(*batch_test)
            # feed
            feed_dict_test = {inputs: x_batch_test, labels: y_batch_test, keep_prob: 1}
            # training run
            summary1, step, _test_loss, _test_accuracy = sess.run([merged, global_step, test_loss, test_accuracy],
                                                            feed_dict=feed_dict_test)
            writer1.add_summary(summary1)
            # data
            time_str1 = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
            # print
            print('----------{}:step{}, test_loss:{:g}, test_acc:{:g}%'.format(time_str1, step, _test_loss, _test_accuracy * 100))
        else:
            # 获取训练数据，字典解压
            x_batch_train, y_batch_train = zip(*batch_train)
            # feed
            feed_dict_train = {inputs: x_batch_train, labels: y_batch_train, keep_prob: drop_keep_prob}
            # training run
            _, summary, step, _train_loss, _train_accuracy = sess.run([train_op, merged, global_step,
                                                                train_loss, train_accuracy], feed_dict=feed_dict_train)
            writer.add_summary(summary)
            # data
            time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
            # print
            print('{}:step{}, train_loss:{:g}, train_acc:{:g}%'.format(time_str, step, _train_loss, _train_accuracy*100))

        if temp % save_every_n == 0:
            saver.save(sess, './model/my_cnn_text_classifier.ckpt', global_step=global_step)
            print('模型在当前{}步骤已保存！'.format(step))
        if temp > max_steps:
            print('超出最大迭代退出！')
            break
    writer.close()
    writer1.close()
    saver.save(sess, './model/my_cnn_text_classifier.ckpt', global_step=global_step)























