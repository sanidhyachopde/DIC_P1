# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
import time
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler


def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """

def Recall(y_true,y_pred):
     """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    K = np.unique(y_true) # Number of classes 
    matrix = np.zeros((len(K), len(K)))
    ima = dict(zip(K, range(len(K))))
    for p, a in zip(y_pred, y_true):
            matrix[ima[a], ima[p]] += 1
    return matrix.astype(dtype=np.int64)
    # also get the accuracy easily with numpy
    #accuracy = (actual == predicted).sum() / float(len(actual))

def KNN(X_train,X_test,Y_train , k):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.sqrt((X_test**2).sum(axis=1)[:, np.newaxis] + (X_train**2).sum(axis=1) - 2 * X_test.dot(X_train.T))
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      #closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # training point, and use self.y_train to find the labels of these      #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      y_indicies = np.argsort(dists[i, :], axis = 0)
      closest_y = Y_train[y_indicies[:k]]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = np.argmax(np.bincount(closest_y))
    return y_pred   
    
def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    
def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """

def SklearnSupervisedLearning(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    predictions = []
    X_train = MinMaxScaler().fit_transform(X_train)
    
    #SVM Implementation
    sv = svm.SVC(kernel='poly', gamma=2)
#    lab_enc = preprocessing.LabelEncoder()
#    training_scores_encoded = lab_enc.fit_transform(Y_train)    
#    sv.fit(X_train, training_scores_encoded)     
    sv.fit(X_train, Y_train)
    svPrediction = sv.predict(X_test)
    predictions.append(svPrediction)
    print("SVM Prediction: ", svPrediction)
    print("SVM Accuracy: ", sv.score(X_train, Y_train))
    
    #Logisting Regression Implementation
    lr = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')
    lr.fit(X_train, Y_train)
    lrPrediction = lr.predict(X_test)
    predictions.append(lrPrediction)
    print("LR Prediction: ",lrPrediction)
    print("LR Accuracy: ", lr.score(X_train, Y_train))

    #Decision Tree Implementation
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(X_train,Y_train)
    dtPrediction = dt.predict(X_test)
    predictions.append(dtPrediction)
    print("Decision Tree Prediction: ",dtPrediction)
    print("DT Accuracy: ", dt.score(X_train, Y_train))
    
    #KNN Implementation
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    knnPrediction = knn.predict(X_test)
    predictions.append(knnPrediction)
    print("KNN Prediction: ",knnPrediction)
    print("KNN Accuracy: ", knn.score(X_train, Y_train))
        
    endTime = time.time()
    print ((endTime-startTime)/60)
    
    return predictions

def SklearnVotingClassifier(X_train,Y_train,X_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """    
    X_train = MinMaxScaler().fit_transform(X_train)
    
    sv = svm.SVC(kernel='poly', gamma=2)
    lr = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')
    dt = tree.DecisionTreeClassifier()
    knn = KNeighborsClassifier(n_neighbors=3)
    
    models.append(('svm', sv))
    models.append(('logistic', lr))
    models.append(('decisiontree', dt))
    models.append(('kneighborsClassifier', knn))
    
    votingensemblemodel = VotingClassifier(models)
    votingensemblemodel = votingensemblemodel.fit(X_train, Y_train)
    vcAccuracy = votingensemblemodel.score(X_train, Y_train)
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
#X = (X - np.min(X))/(np.max(X) - np.min(X)) #normalising
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
temp = KNN(X_train,X_test,Y_train , 3)
temp = temp.astype(np.int64)
count = 0
conf = ConfusionMatrix(Y_test,temp)
print(conf)
for i in range(temp.shape[0]):
    if temp[i]==Y_test[i]:
        count+=1
print(count/temp.shape[0])
startTime = time.time()
models = []
predictions = SklearnSupervisedLearning(X_train,Y_train,X_test)
votingClassifierPrediction = SklearnVotingClassifier(X_train,Y_train,X_test)
print("")
print(predictions)
print("")
confusionMatrixVisualization(Y_test, predictions)


        



    
