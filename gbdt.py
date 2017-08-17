#/usr/bin/python3

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

X = np.array([
    [1,],
    [2,],
    [3,],
    [4,],

    ], dtype=np.float64)
y = np.array([.1, .2, .3, .4])


class MyDecisionTreeRegressor(object):
    """docstring for MyDecisionTreeRegressor"""
    def __init__(self, max_depth=5, num_split=10):
        super(MyDecisionTreeRegressor, self).__init__()
        self.j_li = []
        self.s_li = []
        self.c_li = []
        self.max_depth = max_depth
        self.num_split = num_split

    def min_index_2darray(array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] == array.min():
                    return i, j
        return -1, -1
        
    def calc_j_s_c(self, X, y):
        assert X.shape[0] == y.shape[0]
        loss = np.zeros((X.shape[1], self.num_split))
        s = np.zeros((X.shape[1], self.num_split))
        c = np.zeros((X.shape[1], self.num_split, 2))
        for j in range(X.shape[1]):
            smin = np.min(X[:, j])
            smax = np.max(X[:, j])
            # print(smin, smax)
            s[j] = np.linspace(smin, smax, self.num_split)
            for k in range(self.num_split):
                c1 = []
                c2 = []
                for n in range(X.shape[0]):
                    if X[n][j] < s[j][k]:
                        c1.append(y[n])
                    else:
                        c2.append(y[n])
                c1 = np.mean(c1)
                c2 = np.mean(c2)
                c[j][k] = c1, c2
                loss[j][k] = 0
                for n in range(X.shape[0]):
                    if X[n][j] < s[j][k]:
                        loss[j][k] += (y[n] - c1) ** 2
                    else:
                        loss[j][k] += (y[n] - c2) ** 2

        j, k = MyDecisionTreeRegressor.min_index_2darray(loss)
        # print('calc_j_s_c len(X):{} j:{} k:{} s:{:.4f} loss:{:.4f} split:{:.4f} split:{} c:{}'.format(len(X), j, k, s[j][k], loss[j][k], s[j][k], s, c[j][k]))
        return j, s[j][k], c[j][k]

    def split_dt(X, y, j, s, c):
        X1 = []
        y1 = []
        X2 = []
        y2 = []
        for i in range(X.shape[0]):
            if X[i][j] < s:
                X1.append(X[i])
                y1.append(y[i])
            else:
                X2.append(X[i])
                y2.append(y[i])
        return np.array(X1), np.array(y1), np.array(X2), np.array(y2)

    def fit(self, X, y):

        j, s, c = self.calc_j_s_c(X, y)
        
        self.j_li.append(j)
        self.s_li.append(s)
        self.c_li.append(c)

        X1, y1, X2, y2 = MyDecisionTreeRegressor.split_dt(X, y, j, s, c) 

        # print('debug: j {} s {} c {}'.format(self.j_li, self.s_li, self.c_li))
        if len(y1) >= 2 and len(y2) >= 2 and self.max_depth > 0:
            self.max_depth -= 1
            self.fit(X1, y1)
            self.fit(X2, y2)
            pass

    def predict(self, X):
        assert len(self.j_li) == len(self.s_li) == len(self.c_li)
        y_li = []
        for x in X:
            index = 0
            y = self.c_li[index][0]

            while True:
                if index >= len(self.c_li):
                    break
                j = self.j_li[index]
                s = self.s_li[index]
                y = self.c_li[index][0]

                # print('rich index:{} j:{} s:{} x[j]:{} y:{} c:{}'.format(index, j, s, x[j], y, self.c_li))

                if x[j] < s:
                    # print('left')
                    y = self.c_li[index][0]
                    next_index = index * 2 + 1
                    if next_index >= len(self.c_li):
                        y = self.c_li[index][0]
                        break
                    else:
                        index = next_index
                else:
                    # print('right')
                    y = self.c_li[index][0]
                    next_index = index * 2 + 2
                    if next_index >= len(self.c_li):
                        y = self.c_li[index][1]
                        break
                    else:
                        index = next_index


            y_li.append(y)
        return np.array(y_li)

X = iris.data[:500, :]
y = iris.target[:500]

max_depth = 1

dt = MyDecisionTreeRegressor(max_depth=max_depth, num_split=10)
dt.fit(X, y)
print('my decision tree MSE', np.mean((y - dt.predict(X)) ** 2))

clf = tree.DecisionTreeRegressor(max_depth=max_depth)
clf = clf.fit(X, y)
print('sklearn decision tree MSE', np.mean((y - clf.predict(X)) ** 2))

