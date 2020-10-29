"""
    模型数据读取模块
    1.模型分类：9类(culture,education,finance,health,it,military,recruitment,sport,tourism)
"""

import jieba
import os
import re
import numpy as np
import itertools
from collections import Counter
import json

# 样本数据的存储地址
# base_path = 'D:/my_AI/cnn_sougou_text_classify/'
base_path = 'C:/Users/Administrator/Desktop/工作/文本语料库/sogou/'
classes = ['culture', 'education', 'finance', 'health', 'it', 'military', 'recruitment', 'sport', 'tourism']
padding_word = "<PAD/>"


# 获取数据函数,加载主函数
# 返回：样本，标签，词汇表，逆词汇表
def load_data():
    # sentences, labels类型：list
    sentences, labels = load_data_label()
    # paded_sentences类型：list
    paded_sentences = pad_sentences(sentences)
    # vocabulary类型：字典；vocabulary_inv类型：list
    vocabulary, vocabulary_inv = build_vocab(paded_sentences)
    print(vocabulary)
    x, y = build_input_data(paded_sentences, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


# 单个样本的预测函数
# 思路：取出若干个样本和对应的标签，加载数据的时候要注意
# 直接返回样本和标签
def load_pred_data():
    # 获取数据样本,先用字典里面的数据。如果不用字典里面的数据，新的数据来了就查找不到，会报错。
    pre_sentences, pre_labels = load_data_label(start_index=480, end_index=500)
    # #########################################################################
    pre_paded_sentences = pad_sentences(pre_sentences)
    # 加载训练时刻的字典
    _, _, vocabulary, _ = load_data()
    print(vocabulary)
    # 获取数据
    x, y = build_input_data(pre_paded_sentences, pre_labels, vocabulary)
    return [x, y]


# 加载数据标签（测试有效）
# index：获取数据的索引，就是你要获取多少数据，默认取前500个
# 返回：样本list，标签list
def load_data_label(start_index=0, end_index=500):
    post_list = []
    label_list = []
    # one-hot向量
    labels = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1]]
    # 用于标签的填入
    i = 0
    # 遍历类别标签
    for c in classes:
        # 该目录下的文件
        file_list = os.listdir(base_path + 'sougo_text_data/' + c)
        # 文件太多，根据训练预料获得特定数量的的数据。前500个数据
        file_list_drop = file_list[start_index:end_index]
        for files in file_list_drop:
            # 打开文件
            f = open(base_path + 'sougo_text_data/' + c + '/' + files, 'r', encoding='gbk', errors='ignore')
            temp = f.read().replace('&nbsp', '')
            # 正则替换
            data = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、：；;《》“”~@#￥%……&*（）]+", "", temp)
            # 保存查看
            # f2 = open('./test.txt', 'w')
            # f2.write(data)
            # 分词
            _data = list(jieba.cut(data))
            post_list.append(_data)
            # 提取数据的同时加标签
            label_list.append(labels[i])
            f.close()
        # 类别加一
        i += 1
    return post_list, label_list


# 对句子进行padding,使之具有相同的长度(测试有效）
# 返回：padded 句子
def pad_sentences(sentences):
    # 找到最大长度
    sequences_length = max(len(x) for x in sentences)
    # 新建list
    padded_sentences = []
    # 遍历每个句子
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequences_length - len(sentence)
        new_sentence = sentence + [padding_word]*num_padding
        # 由于时间过程，在这里先做一个截断，取前2000个数据，后期根据需要可以修改
        new_sentence = new_sentence[:10000]
        padded_sentences.append(new_sentence)
    return padded_sentences


# 构建词典
# 输入的sentences必须是paded过的
# 返回：词汇表(字典)，逆词汇表
def build_vocab(sentences):
    # 计数
    word_counts = Counter(itertools.chain(*sentences))
    # 词高频到低频找出来
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # 创建词汇及其索引
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


# 输入的sentences, labels数据都是list，vocabulary是dict，构建输入数据，将每一句话中的词转换成词典中对应的索引。实现了用数字来表示字符，就可以让计算机进行识别
# 返回：样本，标签
def build_input_data(sentences, labels, vocabulary):
    x_data = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y_data = np.array(labels)
    # 返回用[]连接
    return [x_data, y_data]


# 从list对中得到训练的batch,data里面本身就是一个字典
# 返回：迭代目标次产生的batches，在此处batch的内容已经处理好了，直接从中获取即可
def get_batch(data, batch_size, num_epochs):
    # 把数据转成list好处理
    data = list(data)
    data_size = len(data)
    # 每一迭代有多少batch，int()向下取整
    num_batches_per_epoch = int(data_size/batch_size)
    # 对于每一代
    for epoch in range(num_epochs):
        # 做shuffle,返回类似：array([4, 6, 5, 9, 1, 8, 0, 7, 3, 2])
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # 数据已经打乱了
        data_shuffled = np.array(data)[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*batch_size
            end_index = min((batch_num + 1)*batch_size, data_size)
            yield data_shuffled[start_index:end_index]


if __name__ == '__main__':
    # sentences, labels类型：list
    sentences, labels = load_data_label()
    # # paded_sentences类型：list
    # paded_sentences = pad_sentences(sentences)
    # # vocabulary类型：字典；vocabulary_inv类型：list
    # vocabulary, vocabulary_inv = build_vocab(paded_sentences)
    # x, y = build_input_data(paded_sentences, labels, vocabulary)
    # # 将样本和标签打包，好处理
    # batches = get_batch(zip(x, y), 50, 20)
    # for batch in batches:
    #     res = batch
    #     print(batch)
    # load_data()
    # load_pred_data()
    # load_pred_data()
    # load_data_label(start_index=0, end_index=500)

    print('ok')