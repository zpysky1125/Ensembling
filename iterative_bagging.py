# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split


# 读输入，没使用
def read_input(filepath):
    data = []
    labels = []
    with open(filepath) as ifile:
        for line in ifile:
            tokens = line.strip().split('	')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(float(tokens[-1]))
    x = np.array(data)
    y = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x_train, x_test, y_train, y_test


# iterative bagging，本质上就是对残差使用 bagging 的学习器进行学习。最终得到多个 bagging，每个 bagging 对应于残差的估计学习器
def iterative_bagging(base_estimator, x_train, y_train):
    iterative_bagging_estimator = []
    min_mean_square = 1e20
    while True:
        clf = BaggingRegressor(base_estimator=base_estimator, n_estimators=100)
        clf.fit(x_train, y_train)
        y_train_result = clf.predict(x_train)
        y_train = y_train - y_train_result
        iterative_bagging_estimator.append(clf)
        if 1.1*np.sum(y_train**2) > min_mean_square:
            break
        min_mean_square = min(min_mean_square, np.sum(y_train**2))
    return iterative_bagging_estimator


# 预测，每个 bagging 学习器结果相加就可
def predict(iterative_bagging_estimator, x_test, y_test):
    y_test_result = np.zeros(len(x_test))
    for estimator in iterative_bagging_estimator:
        y_test_result = y_test_result + estimator.predict(x_test)
        print r_square(y_test, y_test_result)


# 计算 R square 对结果进行评估
def r_square(y_test, y_test_result):
    sstot = np.sum((y_test - y_test.mean())**2)
    ssreg = np.sum((y_test - y_test_result)**2)
    return 1 - ssreg / sstot


x_train, x_test, y_train, y_test = read_input("airfoil_self_noise.dat")
iterative_bagging_estimator = iterative_bagging(tree.DecisionTreeRegressor(), x_train, y_train)
predict(iterative_bagging_estimator, x_test, y_test)

