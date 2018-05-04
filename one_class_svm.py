"""build a one class SVM to filter the adversarial samples"""
import numpy as np
import os
from sklearn import svm
from keras.datasets import mnist
import random

path = r"/home/zfeng3/cs782/onclass_svm/"

X = np.load(path + "ADsample.npy")
X = X.reshape([20000, -1])
index_random = random.sample(range(20000), 20000)
X = X[index_random]
X_train = X[:15000]

X_test = X[15000:]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_outliers = x_train[:20000]
X_outliers = X_outliers[index_random]
X_outliers = X_outliers.reshape([20000, -1])
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.2)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

print("error in training dataset: {}\n".format(n_error_train))
print("error in testing dataset: {}\n".format(n_error_test))
print("error in outlies dataset: {}\n".format(n_error_outliers))