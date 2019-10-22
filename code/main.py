import xgb


def result():
    """
    生成结果文件的函数
    :return:  void
    """
    train_mat, train_label = xgb.train_loop_file2matrix('../data/data_train.csv')
    test_mat = xgb.test_loop_file2matrix('../data/data_test.csv')
    # 单次循环的最大迭代次数
    max_round = 100000
    # 最大循环次数，不建议取过大的值
    max_loop = 10
    # 当前自我调用次数
    index = 0
    xgb.loop(train_mat, train_label, test_mat, max_round, max_loop, index)


result()
