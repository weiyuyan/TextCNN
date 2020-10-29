"""
    功能：CNN+DNN文本分类
    目前的模型：wordembedding_layer,cnn_layer,full_connected_layer
"""
import tensorflow as tf
import numpy as np
import datetime
from cnn_sougou_text_classify import my_data_helper
from cnn_sougou_text_classify import function_helper

# 超参
sequence_length = 0
classes_num = 9
vocabulary_size = 0
embedding_size = 100
l2_lambda = 0.001
batch_size = 50
epochs = 1000
learning_rate = 0.001
drop_keep_prob = 0.5
filters_height = [2, 3, 4]
filter_num_per_height = [100, 100, 100]

logdir='D:/my_AI/cnn_sougou_text_classify/graph'
# 执行tensorboard：tensorboard --logdir=D:/my_AI/cnn_sougou_text_classify/graph

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
# 确定序列的长度
sequence_length = train_input.shape[1]
print('该训练集中词汇表大小：{:d}'.format(vocabulary_size))
print('一个句子序列的长度为：{:d}'.format(sequence_length))


# 构建数据流图
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, [None, sequence_length], name='inputs')
    with tf.name_scope('labels'):
        train_labels = tf.placeholder(tf.float32, [None, classes_num], name='labels')
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
            embed = tf.nn.embedding_lookup(embeddings, train_inputs, name='embed')
            # 作为卷积的直接输入。卷积要求必须有通道数，虽然文本的厚度为1，只有一个通道，但要加上
            conv_inputs = tf.expand_dims(embed, -1)

        # 卷积层
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

        filter_num_total = sum(filter_num_per_height)
        # 就是平铺,tf.concat(features_pooled, 3):第4个维度进行拼接
        features_pooled_flat = tf.reshape(tf.concat(features_pooled, 3), [-1, filter_num_total])
        # 节点dropout
        features_pooled_flat_drop = tf.nn.dropout(features_pooled_flat, keep_prob=keep_prob, name='drop_out')

        # 第一隐含层
        first_hidden_layer_output, l2_loss = function_helper.nn_layer(features_pooled_flat_drop, filter_num_total, 32,
                                                                      'first_hidden_layer', l2_loss=l2_loss)
        # first_hidden_layer_output_drop = tf.nn.dropout(first_hidden_layer_output, keep_prob=keep_prob,
        #                                                name='first_hidden_drop_out')
        # 第二隐含层，精确度在震荡，有问题
        # second_hidden_layer_output, l2_loss = function_helper.nn_layer(first_hidden_layer_output, 128, 64,
        #                                                                'second_hidden_layer', l2_loss=l2_loss)
        # second_hidden_layer_output_drop = tf.nn.dropout(second_hidden_layer_output, keep_prob=keep_prob,
        #                                                 name='second_hidden_drop_out')

        with tf.name_scope('full_connected_layer'):
            with tf.name_scope('weight'):
                weight = tf.Variable(tf.truncated_normal(shape=[32, classes_num], dtype=tf.float32), name='weight')
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
                scores = tf.nn.xw_plus_b(first_hidden_layer_output, weight, bias, name='xw_plus_b')
                tf.summary.histogram('xw_plus_b', scores)
            # cross_entropy loss
            with tf.name_scope('softmax_cross_entropy'):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_labels, logits=scores, name='losses')
            # loss, is a scalar
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(losses) + l2_lambda * l2_loss
                tf.summary.scalar('loss', loss)
            # 预测
            with tf.name_scope('prediction'):
                predictions = tf.argmax(scores, 1)
                correct_predictions = tf.equal(predictions, tf.argmax(train_labels, 1), name='correct_predictions')
            # accuracy
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
                tf.summary.scalar('accuracy', accuracy)

# 运行数据流图
with tf.Session(graph=graph) as sess:
    # 全局迭代数
    global_step = tf.Variable(0, trainable=False)
    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # train
    with tf.name_scope('train_op'):
        train_op = optimizer.minimize(loss, global_step=global_step)
    sess.run(tf.global_variables_initializer())
    # 获取数据,shuffled过后的数据
    batches = my_data_helper.get_batch(zip(x_shuffled, y_shuffled), batch_size, epochs)
    # 所有汇总
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir=logdir, graph=graph)
    # saver
    saver = tf.train.Saver()

    # 迭代训练
    for batch in batches:
        # 获取数据，字典解压
        x_batch, y_batch = zip(*batch)
        # feed
        feed_dict = {train_inputs: x_batch, train_labels: y_batch, keep_prob: drop_keep_prob}
        # training run
        _, summary, step, _loss, _accuracy = sess.run([train_op, merged, global_step, loss, accuracy], feed_dict=feed_dict)
        # data
        time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
        # print
        print('{}:step{}, loss:{:g}, acc:{:g}%'.format(time_str, step, _loss, _accuracy*100))
        # write summary
        writer.add_summary(summary)
    saver.save(sess, './model/my_cnn_text_classifier.ckpt', global_step=global_step)























