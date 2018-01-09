# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier


# 读输入
def read_input():
    map_word = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
        ,'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'draw': 17}
    data = []
    labels = []
    with open("krkopt.data") as ifile:
        for line in ifile:
            tokens = line.strip().split(',')
            data.append([float(tk) if tk.isdigit() else float(ord(tk)-ord('a')) for tk in tokens[:-1]])
            labels.append(map_word.get(tokens[-1]))
    x = np.array(data)
    y = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x_train, x_test, y_train, y_test


# 简单的决策树
def decision_tree_predict(x_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf("iris.pdf")
    print accuracy_score(y_test, clf.predict(x_test))


# 决策树的bagging
def bagging_decision_tree_predict(x_train, y_train, x_test, y_test):
    meta_clf = tree.DecisionTreeClassifier()
    clf = BaggingClassifier(meta_clf, n_estimators=100)
    clf.fit(x_train, y_train)
    print accuracy_score(y_test, clf.predict(x_test))


# 随机森林
def random_forest_predict(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    print accuracy_score(y_test, clf.predict(x_test))


# 决策树的adaboost实现
def adaboost_decision_tree_predict(x_train, y_train, x_test, y_test):
    meta_clf = tree.DecisionTreeClassifier()
    clf = AdaBoostClassifier(meta_clf, n_estimators=100)
    clf.fit(x_train, y_train)
    print accuracy_score(y_test, clf.predict(x_test))


x_train, x_test, y_train, y_test = read_input()
decision_tree_predict(x_train, y_train, x_test, y_test)
bagging_decision_tree_predict(x_train, y_train, x_test, y_test)
random_forest_predict(x_train, y_train, x_test, y_test)
adaboost_decision_tree_predict(x_train, y_train, x_test, y_test)