#/usr/bin/python3

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

X = np.array([
    [1,1,0], 
    [1,.5,0], 
    [1,.6,0], 
    [1.1,1.2,0], 
    [0,1,1], 
    [0,1.1,1.2], 
    [0,1,2], 
    [0,1.1,2.1], 
    [0,2,3],
    [0,2.1,3.1],
    ], dtype=np.float64)
y = np.array([1, 0.4, 0.9, 1.1, 1.2, 1.3, 1.5, 1.6, 2.0, 2.1])


class MyDecisionTreeRegressor(object):
    """docstring for MyDecisionTreeRegressor"""
    def __init__(self):
        super(MyDecisionTreeRegressor, self).__init__()
        self.j_li = []
        self.s_li = []
        self.c_li = []

    def min_index_2darray(array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] == array.min():
                    return i, j
        return -1, -1
        
    def calc_j_s_c(X, y):
        assert X.shape[0] == y.shape[0]
        NUM_SPLITS = 10
        loss  = np.zeros((X.shape[1], NUM_SPLITS))
        s = np.zeros((X.shape[1], NUM_SPLITS))
        c     = np.zeros((X.shape[1], NUM_SPLITS, 2))
        for j in range(X.shape[1]):
            smin = np.min(X[:, j])
            smax = np.max(X[:, j])
            # print(smin, smax)
            s[j] = np.linspace(smin, smax, NUM_SPLITS)
            for k in range(NUM_SPLITS):
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
        # print('j:{} k:{} loss:{} loss:{}'.format(j, k, loss[j][k], loss))
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

    def fit(self, X, y, max_depth):

        j, s, c = MyDecisionTreeRegressor.calc_j_s_c(X, y)
        
        self.j_li.append(j)
        self.s_li.append(s)
        self.c_li.append(c)

        X1, y1, X2, y2 = MyDecisionTreeRegressor.split_dt(X, y, j, s, c) 

        print('debug: j {} s {} c {}'.format(self.j_li, self.s_li, self.c_li))
        if y1.shape[0] > 2 and y2.shape[0] > 2 and max_depth - 1> 0:
            self.fit(X1, y1, max_depth - 1)
            self.fit(X2, y2, max_depth - 1)
            pass

    def predict(self, X):
        assert len(self.j_li) == len(self.s_li) == len(self.c_li)
        y_li = []
        for x in X:
            depth = 0
            offset = 0
            y = self.c_li[0][0]

            while True:
                index = 0 if depth == 0 else 2 ** depth + offset
                if index >= len(self.j_li):
                    break
                j = self.j_li[index]
                s = self.s_li[index]
                if x[j] < s:
                    # left
                    depth += 1
                    offset = 0
                    y = self.c_li[index][0]
                else:
                    # right
                    depth += 1
                    offset = 1
                    y = self.c_li[index][1]

            y_li.append(y)
        return np.array(y_li)

# X = iris.data
# y = iris.target

max_depth = 2

dt = MyDecisionTreeRegressor()
dt.fit(X, y, max_depth=max_depth)
print('my decision tree MSE', np.mean((y - dt.predict(X)) ** 2))

clf = tree.DecisionTreeRegressor(max_depth=max_depth)
clf = clf.fit(X, y)
print('sklearn decision tree MSE', np.mean((y - clf.predict(X)) ** 2))

