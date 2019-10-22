import xgboost as xgb
import numpy as np
import datetime
import kNN


def test_loop_file2matrix(filename):
    """
    读取文件到矩阵
    :param filename:文件地址
    :return: loop函数用的的test_mat
    """
    fr = open(filename)
    number_lines = len(fr.readlines())
    return_mat = np.zeros((number_lines, 6))
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split(',')
        return_mat[index, :] = list_from_line[1:7]
        index += 1
    return return_mat


def test_last_file2matrix(filename):
    """
    读取文件到矩阵
    :param filename:文件地址
    :return:last函数用的test_mat和label
    """
    fr = open(filename)
    number_lines = len(fr.readlines())
    return_mat = np.zeros((number_lines, 6))
    label_mat = np.zeros((number_lines, 2))
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split(',')
        return_mat[index, :] = list_from_line[1:7]
        label_mat[index, 0] = list_from_line[0]
        label_mat[index, 1] = -1
        index += 1
    return return_mat, label_mat


def train_loop_file2matrix(filename):
    """
    读取文件到矩阵
    :param filename:文件地址
    :return: loop函数用的train_mat和label
    """
    fr = open(filename)
    number_lines = len(fr.readlines())
    return_mat = np.zeros((number_lines, 6))
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


def last(train_mat, train_label, test_mat, test_label_csv):
    """
    最后一次的函数
    :param train_mat: 训练数据
    :param train_label: 训练数据标签
    :param test_mat: 测试数据
    :param test_label_csv: 测试数据CSV，只存储id和label
    :return:
    """
    data_train = xgb.DMatrix(train_mat, label=train_label)
    data_test = xgb.DMatrix(test_mat, label=[])
    watch_list = [(data_train, 'train')]

    # 设置参数
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    """
        max_depth : 每一棵树的最大深度，2:的意思：1个节点，分2个，再分4个。这边是深度为2
        eta : 防止过拟合参数。
        silent : 是否输出计算过程。0否则不silent，输出计算过程
        objective : 需要做什么。binary 表示 二分类\
            logitraw 是从负无穷到正无穷，后续代码判断y_hat，y_hat应该大于0
            logistic 是从0到1，后续代码判断y_hat，y_hat应该大于0.5
            在这里只能用logistic，因为后续需要根据logistic对数据进行修正
    """
    n_round = 1000000
    """
        n_round : 迭代次数，次数高的话错误率可能会少
    """
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watch_list)
    # 计算错误率
    y_hat = bst.predict(data_test)
    print(y_hat)
    m = test_mat.shape[0]
    # print(test_mat)
    # print(test_label_csv)
    # modify k 修正参数K
    modify_k = 0.1
    modify = 0
    for i in range(m):
        label = (y_hat[i] > 0.5)
        if (np.abs(y_hat[i] - 0.5)) < modify_k:
            modify = modify + 1
            # print(test_mat[m - 1])
            # 用kNN 进行修正
            # print('修正前' + str(label))
            # label = kNN.result(test_mat[i])
            # print('修正后' + str(label))

            # 反转
            if label == 0:
                label = 1
            else:
                label = 0
        test_label_csv[i, 1] = label
    np.savetxt('../submit/result' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv', test_label_csv,
               delimiter=',', fmt='%d')
    print('-' * 60)
    print(test_label_csv)
    print('-' * 60)
    print(modify)
    print('#' * 60)


def loop(train_mat, train_label, test_mat, max_round, max_loop, index):
    """
    循环函数
    :param train_mat: 训练数据
    :param train_label: 训练标签
    :param test_mat: 测试数据
    :param max_round: 单次循环的最大迭代次数
    :param max_loop: 最大循环次数
    :param index: 循环次数标签
    :return: void，最后调用 last
    """
    data_train = xgb.DMatrix(train_mat, label=train_label)
    data_test = xgb.DMatrix(test_mat, label=[])
    watch_list = [(data_train, 'train')]

    # 设置参数
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    """
        max_depth : 每一棵树的最大深度，2:的意思：1个节点，分2个，再分4个。这边是深度为2。一般去2至10
        eta : 防止过拟合参数。
        silent : 是否输出计算过程。0否则不silent，输出计算过程
        objective : 需要做什么。binary 表示 二分类
            logitraw 是从负无穷到正无穷，后续代码判断y_hat，y_hat应该大于0
            logistic 是从0到1，后续代码判断y_hat，y_hat应该大于0.5
            在这里只能用logistic，因为后续需要用logistic找到程序有最大把握的数据
    """
    n_round = max_round
    """
        n_round : 迭代次数，次数高的话错误率可能会少
    """
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watch_list)

    y_hat = bst.predict(data_test)
    test_m = test_mat.shape[0]
    confirm_num = 0
    train_mat_temp = []
    train_label_temp = []
    test_loop_mat_temp = []
    for i in range(test_m):
        # 如果绝对值等于0.5
        if (np.abs(y_hat[i] - 0.5)) == 0.5:
            confirm_num = confirm_num + 1
            label = (y_hat[i] > 0.5)
            train_mat_temp.append(test_mat[i])
            train_label_temp.append(label)
        # 如果绝对值不等于0.5
        else:
            test_loop_mat_temp.append(test_mat[i])

    if (index < max_loop) and (confirm_num > 0) and len(test_loop_mat_temp) > 0:
        train_mat = np.vstack((train_mat, np.array(train_mat_temp)))
        train_label = np.hstack((train_label, train_label_temp))
        # 添加 if len(test_loop_mat_temp) > 0,else 的话 调用 main_last
        test_mat = np.array(test_loop_mat_temp)
        index = index + 1
        print('=' * 60)
        print('=' * 60)
        print('=' * 60)
        print('自我调用 ' + str(index) + ' 新增样本 ' + str(confirm_num) + ' 剩余测试数据 ' + str(len(test_loop_mat_temp)))
        print('自我调用 ' + str(index) + ' 新增样本 ' + str(confirm_num) + ' 剩余测试数据 ' + str(len(test_loop_mat_temp)))
        print('自我调用 ' + str(index) + ' 新增样本 ' + str(confirm_num) + ' 剩余测试数据 ' + str(len(test_loop_mat_temp)))
        print('自我调用 ' + str(index) + ' 新增样本 ' + str(confirm_num) + ' 剩余测试数据 ' + str(len(test_loop_mat_temp)))
        print('自我调用 ' + str(index) + ' 新增样本 ' + str(confirm_num) + ' 剩余测试数据 ' + str(len(test_loop_mat_temp)))
        print('自我调用 ' + str(index) + ' 新增样本 ' + str(confirm_num) + ' 剩余测试数据 ' + str(len(test_loop_mat_temp)))
        print('=' * 60)
        print('=' * 60)
        print('=' * 60)
        loop(train_mat, train_label, test_mat, max_round, max_loop, index)
    else:
        test_mat, test_label_csv = test_last_file2matrix('../data/data_test.csv')
        last(train_mat, train_label, test_mat, test_label_csv)
