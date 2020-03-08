# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
import time
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler

from math import log, sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def Accuracy(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float

    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy*100


def Recall(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    true_pos = np.diag(cm)
    recall = np.sum(true_pos / np.sum(cm, axis=1))
    return recall/len(cm[0])


def Precision(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true, y_pred)
    true_pos = np.diag(cm)
    precision = np.sum(true_pos / np.sum(cm, axis=0))
    return precision/len(cm[0])


def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """


def ConfusionMatrix(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    K = np.unique(y_true)  # Number of classes
    matrix = np.zeros((len(K), len(K)))
    ima = dict(zip(K, range(len(K))))
    for p, a in zip(y_pred, y_true):
        matrix[ima[a], ima[p]] += 1
    return matrix.astype(dtype=np.int64)


def KNN(X_train, X_test, Y_train, k):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: numpy.ndarray
    """
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.sqrt((X_test**2).sum(axis=1)
                    [:, np.newaxis] + (X_train**2).sum(axis=1) - 2 * X_test.dot(X_train.T))
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        y_indicies = np.argsort(dists[i, :], axis=0)
        closest_y = Y_train[y_indicies[:k]]
        y_pred[i] = np.argmax(np.bincount(closest_y))
    return y_pred


def RandomForest(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: numpy.ndarray
    """
    class TreeEnsemble():
        def __init__(self, x, y, n_trees, sample_sz, min_leaf=5):
            np.random.seed(42)
            self.x = x
            self.y = y
            self.n_trees = n_trees
            self.sample_sz = sample_sz
            self.min_leaf = min_leaf
            self.trees = [self.create_tree() for i in range(n_trees)]

        def create_tree(self):
            rnd_idxs = np.random.permutation(len(self.x))[:self.sample_sz]
            return DecisionTree(self.x[rnd_idxs], self.y[rnd_idxs], min_leaf=self.min_leaf)

        def predict(self, x):
            return np.mean([t.predict(x) for t in self.trees], axis=0)

    class DecisionTree():
        def __init__(self, x, y, idxs=None, min_leaf=5):
            if idxs is None:
                idxs = np.arange(len(y))

            self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
            self.n, self.c = len(idxs), x.shape[1]

            self.val = np.mean(y[idxs])
            self.score = float('inf')

            self.find_varsplit()

        def find_varsplit(self):
            for var_idx in range(self.c):
                self.find_better_split(var_idx)
            # Ready to split
            if self.is_leaf:
                return
            x = self.split_col
            lhs = np.nonzero(x <= self.split_on)[0]
            rhs = np.nonzero(x > self.split_on)[0]

            self.lhs = DecisionTree(
                self.x, self.y, idxs=self.idxs[lhs], min_leaf=self.min_leaf)
            self.rhs = DecisionTree(
                self.x, self.y, idxs=self.idxs[rhs], min_leaf=self.min_leaf)

        def find_better_split_slow(self, var_idx):
            x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]

            for val in x:
                rhs = val <= x
                lhs = val > x

                if lhs.sum() == 0:
                    continue
                # By Std
                lhs_std = y[lhs].std()
                rhs_std = y[rhs].std()
                curr_score = lhs_std*lhs.sum() + rhs_std*rhs.sum()

                # Using Mse
                lhs_mse = mean_squared_error(
                    y[lhs], [y[lhs].mean() for i in range(y[lhs].shape[0])])
                rhs_mse = mean_squared_error(
                    y[rhs], [y[rhs].mean() for i in range(y[rhs].shape[0])])
                # curr_score = np.array([lhs_mse,rhs_mse]).mean()

                if curr_score < self.score:
                    self.var_idx, self.score, self.split_on, self.var_name = var_idx, curr_score, val, self.x.columns[
                        var_idx]

        def std_fast(self, cnt, s1, s2): return sqrt((s2/cnt) - (s1/cnt)**2)

        def find_better_split(self, var_idx):
            x, y = self.x[self.idxs, var_idx], self.y[self.idxs]

            sort_idx = np.argsort(x)
            sort_x, sort_y = x[sort_idx], y[sort_idx]
            rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
            lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.

            # print(self.n - self.min_leaf-1)
            for i in range(self.n-self.min_leaf-1):
                xi, yi = sort_x[i], sort_y[i]
                lhs_cnt += 1
                rhs_cnt -= 1
                lhs_sum += yi
                rhs_sum -= yi
                lhs_sum2 += yi**2
                rhs_sum2 -= yi**2

                if i < self.min_leaf or xi == sort_x[i+1]:
                    continue

                lhs_std = self.std_fast(lhs_cnt, lhs_sum, lhs_sum2)
                rhs_std = self.std_fast(rhs_cnt, rhs_sum, rhs_sum2)

                curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
                if curr_score < self.score:
                    self.var_idx, self.score, self.split_on, self.var_name = var_idx, curr_score, xi, self.x[
                        var_idx]

        def predict(self, x):
            return np.array([self.predict_row(xi) for xi in x])

        def predict_row(self, xi):
            if self.is_leaf:
                return self.val
            t = self.lhs if xi[self.var_idx] <= self.split_on else self.rhs
            return t.predict_row(xi)

        @property
        def split_col(self): return self.x[self.idxs, self.var_idx]

        @property
        def is_leaf(self): return self.score == float('inf')

        def __repr__(self):
            s = f'n: {self.n}; val: {self.val}'
            if not self.is_leaf:
                s += f' score: {self.score}; split_on: {self.split_on}; var: {self.var_name}'
            return s

    model = TreeEnsemble(X_train, Y_train, 3, 5000, 10)
    y_pred = model.predict(X_test)
    return y_pred


def PCA(X_train, N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """


def Kmeans(X_train, N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """


def SklearnSupervisedLearning(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: List[numpy.ndarray]
    """
    predictions = []
    X_train = MinMaxScaler().fit_transform(X_train)

    # SVM Implementation
    sv = svm.SVC(kernel='poly', gamma=2)
#    lab_enc = preprocessing.LabelEncoder()
#    training_scores_encoded = lab_enc.fit_transform(Y_train)
#    sv.fit(X_train, training_scores_encoded)
    sv.fit(X_train, Y_train)
    svPrediction = sv.predict(X_test)
    predictions.append(svPrediction)
    print("SVM Prediction: ", svPrediction)
    print("SVM Accuracy: ", sv.score(X_train, Y_train))

    # Logisting Regression Implementation
    lr = LogisticRegression(random_state=0, solver='saga',
                            multi_class='multinomial')
    lr.fit(X_train, Y_train)
    lrPrediction = lr.predict(X_test)
    predictions.append(lrPrediction)
    print("LR Prediction: ", lrPrediction)
    print("LR Accuracy: ", lr.score(X_train, Y_train))

    # Decision Tree Implementation
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(X_train, Y_train)
    dtPrediction = dt.predict(X_test)
    predictions.append(dtPrediction)
    print("Decision Tree Prediction: ", dtPrediction)
    print("DT Accuracy: ", dt.score(X_train, Y_train))

    # KNN Implementation
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    knnPrediction = knn.predict(X_test)
    predictions.append(knnPrediction)
    print("KNN Prediction: ", knnPrediction)
    print("KNN Accuracy: ", knn.score(X_train, Y_train))

    endTime = time.time()
    print((endTime-startTime)/60)

    return predictions


def SklearnVotingClassifier(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray

    :rtype: List[numpy.ndarray]
    """
    X_train = MinMaxScaler().fit_transform(X_train)

    sv = svm.SVC(kernel='poly', gamma=2)
    lr = LogisticRegression(random_state=0, solver='saga',
                            multi_class='multinomial')
    dt = tree.DecisionTreeClassifier()
    knn = KNeighborsClassifier(n_neighbors=3)

    models.append(('svm', sv))
    models.append(('logistic', lr))
    models.append(('decisiontree', dt))
    models.append(('kneighborsClassifier', knn))

    votingensemblemodel = VotingClassifier(models)
    votingensemblemodel = votingensemblemodel.fit(X_train, Y_train)
    vcAccuracy = votingensemblemodel.score(
        X_train, Y_train, sample_weight=None)
    vcPrediction = votingensemblemodel.predict(X_test)
    print("VC Prediction: ", vcPrediction)
    print("VC accuracy: ", vcAccuracy)
    return vcPrediction


"""
Create your own custom functions for Matplotlib visualization of hyperparameter search.
Make sure that plots are labeled and proper legends are used
"""


def confusionMatrixVisualization(Y_test, predictions):

    for Y_pred in predictions:
        cm = ConfusionMatrix(Y_test, Y_pred)
        plt.matshow(cm)
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')


data = pd.read_csv("data.csv")
X = data.iloc[:, :48].values
Y = data.iloc[:, 48].values
# X = (X - np.min(X))/(np.max(X) - np.min(X)) #normalising
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
start_time = time.time()
y_pred = RandomForest(X_train, Y_train, X_test)
y_pred = np.rint(y_pred)
y_pred = y_pred.astype(dtype=np.int64)
acc = Accuracy(Y_test, y_pred)
conf = ConfusionMatrix(Y_test, y_pred)
pre = Precision(Y_test, y_pred)
rec = Recall(Y_test, y_pred)
print('Manual')
print(conf)

startTime = time.time()
models = []
predictions = SklearnSupervisedLearning(X_train, Y_train, X_test)
votingClassifierPrediction = SklearnVotingClassifier(X_train, Y_train, X_test)
print("")
print(predictions)
print("")
confusionMatrixVisualization(Y_test, predictions)


print(acc)
print(pre)
print(rec)
print('Sklearn')
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
# pc = PCA(X_train, 2)
# pca = decomposition.PCA(n_components=2)
# pca.fit(X_train)
# res = pca.transform(X_train)
print("--- %s seconds ---" % (time.time() - start_time))
# temp = KNN(X_train,X_test,Y_train , 3)
# temp = temp.astype(np.int64)
# conf = ConfusionMatrix(Y_test,temp)
# print(conf)
