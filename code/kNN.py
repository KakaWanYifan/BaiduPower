from numpy import *
import operator


def file2matrix(filename):
    """
    读取文件到矩阵
    :param filename:
    :return: retur_mat   返回的矩阵
    :return: category_label_vector   类别vector
    """
    fr = open(filename)
    number_lines = len(fr.readlines())
    return_mat = zeros((number_lines, 6))
    return_label = []
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split(',')
        return_mat[index, :] = list_from_line[1:7]
        return_label.append(int(float(list_from_line[-1])))
        index += 1
    return return_mat, return_label


def classify_o(in_x, data_set, labels, k):
    """
    基于欧几里得分类
    :param in_x:需要分类的数据
    :param data_set:分类所参考的数据
    :param labels:分类所参考的数据所对应的标签
    :param k:k为100即选取前100个里面标签数最多的种类，k为10即选取前10个里面标签数最多的种类
    :return:返回需要分类数据的标签
    """
    # 获取data_set的行数
    data_set_size = data_set.shape[0]
    # 依次相见
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    # 二次方
    sq_diff_mat = diff_mat ** 2
    # 相加
    sq_distances = sq_diff_mat.sum(axis=1)
    # 开根号
    distances = sq_distances ** 0.5
    # 排序
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def result(test_mat):
    train_mat, label = file2matrix('../data/data_train.csv')
    classifier_result = classify_o(test_mat, train_mat, label, 10)
    return classifier_result
