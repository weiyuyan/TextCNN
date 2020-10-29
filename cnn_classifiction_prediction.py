"""
    功能：使用训练出的模型，通过该模型进行样本预测
"""
import tensorflow as tf
from cnn_sougou_text_classify import my_data_helper
import numpy as np
import pickle
import codecs
model_path = 'D:/my_AI/cnn_sougou_text_classify/model/my_cnn_text_classifier.ckpt-6749.meta'
model_name = './model/'


# 读取待测样本
# x, y = my_data_helper.load_pred_data()
# zip_xy = zip(x,y)
# # 存储
# save_file = codecs.open('./prediction/pred_data.pkl', 'wb')
# pickle.dump(zip_xy, save_file)
# save_file.close()
# 加载
load_file = codecs.open('./prediction/pred_data.pkl', 'rb')
temp = pickle.load(load_file)
load_file.close()
zipped_res = list(zip(*temp))
# 取出x数据
x = zipped_res[0]
# 取出y数据
y = zipped_res[1]


shuffle_index = np.random.permutation(np.arange(len(x)))
one_index = (np.random.choice(shuffle_index, size=1))[0]

# 在进行新的文本预测时候，要注意输入的维度要和原有模型保持一致，原来的是(?,10000),那么一个文本进来，要变成形状为(1,10000)的shape，就是shape=[1,10000],第一个维度有一个元素，第二个维度有10000个元素。
x1 = np.array([x[one_index]])
y1 = y[one_index]
print(x1)
print(y1)
# 模型预测
with tf.Session() as sess:
    # 加载元数据,找到流程图
    new_saver = tf.train.import_meta_graph(model_path)
    # 加载ckpt
    new_saver.restore(sess, tf.train.latest_checkpoint(model_name))
    # 获取节点
    target = tf.get_collection('pred_network')[0]
    graph = tf.get_default_graph()
    # 获取placeholder,要注意，不同scope下的名字要一层一层的调用才能找到最终的操作.一定要使用获取该操作后的那个outputs的输出，然后取第一个
    # 必须要有outputs[0]，目前我还不知道这个是啥意思。
    input_x = graph.get_operation_by_name('inputs/inputs').outputs[0]
    keep_p = graph.get_operation_by_name('keep_prob/keep_prob').outputs[0]

    pred_result = sess.run(target, feed_dict={input_x: x1, keep_p: 1.0})
    # 对预测结果进行一个softmax操作
    pred_result_with_softmax = sess.run(tf.nn.softmax(pred_result))
    # 检测预测是否正确,找出最大值的位置
    pred_pos = np.argmax(np.squeeze(pred_result), 0)
    actual_pos = np.argmax(y1, 0)
    print('预测结果向量：{}'.format(pred_result))
    print('实际类别：{}     预测类别：{}'.format(actual_pos, pred_pos))



