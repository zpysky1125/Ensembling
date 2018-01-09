# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


# 普通决策树
def decision_tree_predict(x_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    print accuracy_score(y_test, clf.predict(x_test))


# 手写的Adaboost，因为论文中 Adaboost 和 sklearn 的 adaboost 不一样
# 具体区别为论文中的 Adaboost 中，如果error > 0.5，那么就会重新进行 bootstrap 的 sample，并且重新设置 weights
class AdaBoost(object):
    def __init__(self, base_estimator=None, n_estimator=100, target=0.001, x_train=np.array([]), y_train=np.array([])
                 , x_test=np.array([]), y_test=np.array([])):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimator
        self.target = target
        # adaboost 弱分类器的权重
        self.beta = []
        # adaboost 的多个弱分类器
        self.estimators = []
        # x_train 和 y_train 是输入的训练集
        self.x_train = x_train
        self.y_train = y_train
        # x_test 和 y_test 是输入的测试集
        self.x_test = x_test
        self.y_test = y_test
        # adaboost 的权重
        self.weights = [1]*len(self.x_train)
        self.bootstrap = range(0, len(self.x_train))

    # bootstrap sample
    @staticmethod
    def _bootstrap_sample(length):
        idx = np.random.randint(0, length, size=(length))
        return idx

    # 训练一轮
    def _train_iteration(self, iteration):
        clf = self.base_estimator()
        clf.fit(self.x_train[self.bootstrap], self.y_train[self.bootstrap])
        y_train_result = clf.predict(self.x_train[self.bootstrap])
        errors = (self.y_train[self.bootstrap] != y_train_result)
        error = np.sum(self.weights*errors)/len(self.x_train)
        # 如果误差太大，就重新 bootstrap
        if error > 0.5:
            self.bootstrap = self._bootstrap_sample(len(self.x_train))
            self.weights = [1] * len(self.x_train)
            return
        # 如果误差过小，说明当前 bootstrap 已经分类的足够好，重新 bootstrap 以适应不同的数据集
        elif error < 1e-5:
            self.beta.append(1e-10)
            self.bootstrap = self._bootstrap_sample(len(self.x_train))
            self.weights = [1] * len(self.x_train)
        # 否则，正常 adaboost
        else:
            self.beta.append(np.log((1-error)/error))
            self.weights = [0.5*weight/error if errors[index] else 0.5*weight/(1-error)
                            for index, weight in enumerate(self.weights)]
            self.weights = [1e-8 if weight < 1e-8 else weight for weight in self.weights]
        self.estimators.append(clf)

    # 整个训练过程。因为是多分类问题，所以使用投票法
    def train(self):
        for i in range(self.n_estimators):
            self._train_iteration(i)
        result = []
        for i in range(len(self.x_test)):
            result.append([])
        # 统计不同分类器针对的分类结果
        for index, estimator in enumerate(self.estimators):
            y_test_result = estimator.predict(self.x_test)
            for index2, res in enumerate(result):
                res.append([y_test_result[index2], np.log(1/self.beta[index])])
        final_result = []
        # 投票得出结果
        for res in result:
            dic = {}
            for r in res:
                dic[r[0]] = r[1] if not dic.has_key(r[0]) else dic.get(r[0]) + r[1]
            final_result.append(sorted(dic, key=lambda x:dic[x])[-1])
        print float(np.sum(final_result == self.y_test)) / len(self.y_test)


# MultiBoost 的实现，主要区别在于需要设置停止 iteration，在论文的 Table3 中，设置前 √T 个停止 iteration 为 i*√T，主要是为了平均
# 论文中提出使用 poisson 分布来代替 bootstrap，本质上体现不出对 adaboost 的 bagging
# 同时，因为 poisson 分布会聚集在均值周围，所以我认为效果并不会比 bootstrap 更好。这里 poisson 和 bootstrap 都实现了
class MultiBoost(AdaBoost):
    def __init__(self, base_estimator=None, n_estimator=100, target=0.001, x_train=np.array([]), y_train=np.array([])
                 , x_test=np.array([]), y_test=np.array([])):
        super(MultiBoost, self).__init__(base_estimator, n_estimator, target, x_train, y_train, x_test, y_test)
        self.iterations = []
        self.current_iteration = 0
        self._set_iterations()

    # sample from poisson
    @staticmethod
    def _poisson_sample(length):
        bootstrap = []
        for i in range(length):
            tmp = length+1
            while tmp >= length:
                tmp = np.random.poisson(i, 1)
            bootstrap.append(tmp[0])
        return bootstrap

    # 设置停止 iteration
    def _set_iterations(self):
        n = int(float(self.n_estimators)**0.5)
        for i in range(n):
            self.iterations.append(int(((i+1)*self.n_estimators + n - 1)/n))
        for i in range(self.n_estimators):
            self.iterations.append(self.n_estimators)

    def _train_iteration(self, iteration):
        # 体现bagging的一步。如果当前 iteration 等于停止 iteration，说明当前 bagging 应该结束了，就重新采样，同时设置权重
        if self.iterations[self.current_iteration] == iteration:
            self.bootstrap = self._bootstrap_sample(len(self.x_train))
            self.weights = [1] * len(self.x_train)
            self.current_iteration += 1
        clf = self.base_estimator()
        clf.fit(self.x_train[self.bootstrap], self.y_train[self.bootstrap])
        y_train_result = clf.predict(self.x_train[self.bootstrap])
        errors = (self.y_train[self.bootstrap] != y_train_result)
        error = np.sum(self.weights*errors)/len(self.x_train)
        # 如果误差太大，重新 sample
        if error > 0.5:
            self.bootstrap = self._bootstrap_sample(len(self.x_train))
            self.weights = [1] * len(self.x_train)
            self.current_iteration += 1
            return
        # 如果误差太小，说明对当前 bootstrap 的数据集，我们表现的足够好，重新 sample
        elif error < 1e-5:
            self.beta.append(1e-10)
            self.bootstrap = self._bootstrap_sample(len(self.x_train))
            self.weights = [1] * len(self.x_train)
            self.current_iteration += 1
        # 否则，常规 adaboost
        else:
            self.beta.append(np.log((1-error)/error))
            self.weights = [0.5*weight/error if errors[index] else 0.5*weight/(1-error)
                            for index, weight in enumerate(self.weights)]
            self.weights = [1e-8 if weight < 1e-8 else weight for weight in self.weights]
        self.estimators.append(clf)


x_train, x_test, y_train, y_test = read_input()
decision_tree_predict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
adaboost_estimator = AdaBoost(base_estimator=tree.DecisionTreeClassifier, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
adaboost_estimator.train()
multiboost_estimator = MultiBoost(base_estimator=tree.DecisionTreeClassifier, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
multiboost_estimator.train()




