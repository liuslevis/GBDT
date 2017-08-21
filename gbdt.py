#/usr/bin/python3

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import weakref

STOP_NUM = -999999

class Node(object):
    """docstring for Node"""
    def __init__(self, j, s, c, root=None, left=None, right=None):
        super(Node, self).__init__()
        self.j = j
        self.s = s
        self.c = c
        self.root = root #weakref.ref(root) if root else None
        self.left = left
        self.right = right
        
    def count(self):
        left_cnt  = self.left.count()  if self.left  else 0
        right_cnt = self.right.count() if self.right else 0
        return 1 + left_cnt + right_cnt

    def depth(self):
        q = [self.root]
        max_depth = 0
        while True:
            if len(q) == 0:
                break
            p = []
            for node in q:
                if node == self:
                    break
                if node and node.left:
                    p.append(node.left)
                if node and node.right:
                    p.append(node.right)
            q = p
            max_depth += 1

        return max_depth

    def depth_to_leaf(self):
        left_cnt  = self.left.depth()  if self.left  else 0
        right_cnt = self.right.depth() if self.right else 0
        return 1 + max(left_cnt, right_cnt)

    def print(self, indent=0):
        s = '  ' * indent
        s += 'j:{} s:{} c:{}'.format(self.j, self.s, self.c)
        print(s)
        if self.left:
            self.left.print(indent + 1)
        if self.right:
            self.right.print(indent + 1)

    def print_mid_order(self):
        if self.left:
            self.left.print_mid_order()

        print(self.s)

        if self.right:
            self.right.print_mid_order()


class MyDecisionTreeRegressor(object):
    """docstring for MyDecisionTreeRegressor"""
    def __init__(self, max_depth=5, num_split=10):
        super(MyDecisionTreeRegressor, self).__init__()
        self.root = None
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

    def fit(self, X, y, curNode=None, position=None):
        j, s, c = self.calc_j_s_c(X, y)

        newNode = Node(j, s, c, self.root)

        if not self.root:
            self.root = newNode
        elif position == 'left':
            curNode.left = newNode
        elif position == 'right':
            curNode.right = newNode

        X1, y1, X2, y2 = MyDecisionTreeRegressor.split_dt(X, y, j, s, c) 

        # print('debug: depth {} j{} y1 {} y2 {} '.format(newNode.depth(), s, y1, y2))
        if len(y1) >= 2 and len(y2) >= 2 and newNode and newNode.depth() <= self.max_depth:
            self.fit(X1, y1, curNode=newNode, position='left')
            self.fit(X2, y2, curNode=newNode, position='right')

    def predict(self, X):
        assert self.root
        y_li = []
        for x in X:
            index = 0
            curNode = self.root
            y = curNode.c[0]
            while True:
                if x[curNode.j] < curNode.s:
                    if curNode.left:
                        curNode = curNode.left
                    else:
                        y = curNode.c[0]
                        break
                else:
                    if curNode.right:
                        curNode = curNode.right
                    else:
                        y = curNode.c[1]
                        break
            y_li.append(y)
        return np.array(y_li)

    def print_mid_order(self):
        self.root.print_mid_order()


class MyBoostingDecisionTreeRegressor(object):
    def __init__(self, num_tree=3, max_depth=5, num_split=10):
        super(MyBoostingDecisionTreeRegressor, self).__init__()
        self.max_depth = max_depth
        self.num_split = num_split
        self.num_tree = num_tree
        self.trees = []

    def fit(self, X, y):
        # T1
        t0 = MyDecisionTreeRegressor(self.max_depth, self.num_split)
        t0.fit(X, y)
        self.trees.append(t0)
        residual = y - t0.predict(X)

        for i in range(self.max_depth - 1):
            t1 = self.trees[i]
            t2 = MyDecisionTreeRegressor(self.max_depth, self.num_split)
            t2.fit(X, residual)
            self.trees.append(t2)
            residual -= t2.predict(X)

    def predict(self, X):
        Y = np.zeros(X.shape[0])
        for ti in self.trees:
            Y += ti.predict(X)
        return Y

class MyGBDTRegressor(MyBoostingDecisionTreeRegressor):
    """docstring for MyGBDTRegressor
    When loss is mean square loss, the GBDT are in form of Boosting Decision Tree

    """
    def __init__(self, num_tree=3, max_depth=5, num_split=10):
        super(MyGBDTRegressor, self).__init__()
        self.max_depth = max_depth
        self.num_split = num_split
        self.num_tree = num_tree
        self.trees = []


iris = load_iris()

n = 10
X = np.array([[i] for i in range(n)])
y = np.array([i for i in range(n)])

# X = iris.data[0:130,:]
# y = iris.target[0:130]

max_depth = 3
num_tree = 10
num_split = 100
dt = MyDecisionTreeRegressor(max_depth=max_depth, num_split=num_split)
dt.fit(X, y)
# dt.root.print()
# print(dt.predict(X))
print('MyDecisionTreeRegressor MSE', np.mean((y - dt.predict(X)) ** 2))

bdt = MyBoostingDecisionTreeRegressor(num_tree=num_tree, max_depth=max_depth, num_split=num_split)
bdt.fit(X, y)
print('MyBoostingDecisionTreeRegressor MSE', np.mean((y - bdt.predict(X)) ** 2))

gbdt = MyGBDTRegressor(num_tree=num_tree, max_depth=max_depth, num_split=num_split)
gbdt.fit(X, y)
print('MyGBDT MSE', np.mean((y - gbdt.predict(X)) ** 2))

clf = tree.DecisionTreeRegressor(max_depth=max_depth)
clf = clf.fit(X, y)
# print(clf.predict(X))
print('sklearn.DecisionTreeRegressor MSE', np.mean((y - clf.predict(X)) ** 2))

