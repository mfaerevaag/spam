import numpy as np
import csv
import pprint
import operator
from pylab import *

from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as pltn

# load libs
import sys
sys.path.insert(0, "../../lib")
from toolbox import feature_selector_lr, bmplot

# pretty printer
pp = pprint.PrettyPrinter(indent=8)

file = "../../data/spambase.data"

names = [
    'make', 'address', 'all',
    '3d', 'our', 'over',
    'remove', 'internet', 'order',
    'mail', 'receive', 'will',
    'people', 'report', 'addresses',
    'free', 'business', 'email',
    'you', 'credit', 'your',
    'font', '000', 'money',
    'hp', 'hpl', 'george',
    '650', 'lab', 'labs',
    'telnet', '857', 'data',
    '415', '85', 'technology',
    '1999', 'parts', 'pm',
    'direct', 'cs', 'meeting',
    'original', 'project', 're',
    'edu', 'table', 'conference',
    ';', '(', '[', '!', '$', '#',
    'capital_run_length_average',
    'capital_run_length_longest',
    'capital_run_length_total' ]


data = []

# read data
f = open(file)
reader = csv.reader(f)
next(reader, None)
for row in reader: data.append(row)
f.close()

# convert to numpy arrays (last col is y)
X = np.array([x[:-1] for x in data]).astype(np.float)
y = np.array([x[-1] for x in data]).astype(np.float)

N, M = X.shape

# standardize
# X = (X - X.mean()) / (X.max() - X.min())


def top_features(n, weights):
    return sorted(zip(names, weights), reverse=True, key=operator.itemgetter(1))[:n]


# Naive Bayes classifier
def clf_naive_bayes():
    alpha = 1.0        # additive parameter (e.g. Laplace correction)
    est_prior = True   # uniform prior (change to True to estimate prior from data)

    # K-fold crossvalidation
    K = 10
    CV = cross_validation.KFold(N, K, shuffle=True)

    errors = np.zeros(K)
    k=0
    for train_index, test_index in CV:
        # print('Crossvalidation fold: {0}/{1}'.format(k + 1, K))

        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=RandomState())

        clf = MultinomialNB(alpha=alpha, fit_prior=est_prior)
        clf.fit(X_train, ravel(y_train))
        y_est_prob = clf.predict_proba(X_test)
        y_est = np.argmax(y_est_prob,1)

        errors[k] = np.sum(y_est.ravel()!=y_test.ravel(),dtype=float)/y_test.shape[0]
        k+=1

    # Plot the classification error rate
    print('Naive Bayes (multinomial):')
    print('\t error rate: %0.2f%%' % (100 * mean(errors)))
    print('\t top 5 features:')
    pp.pprint(top_features(5, clf.feature_log_prob_[1,:]))


# Decision Tree
def clf_dtree():
    alpha = 1.0        # additive parameter (e.g. Laplace correction)
    est_prior = True   # uniform prior (change to True to estimate prior from data)

    # K-fold crossvalidation
    K = 10
    CV = cross_validation.KFold(N, K, shuffle=True)

    errors = np.zeros(K)
    k=0
    for train_index, test_index in CV:
        # print('Crossvalidation fold: {0}/{1}'.format(k + 1, K))

        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]

        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(X_train, ravel(y_train))
        y_est_prob = clf.predict_proba(X_test)
        y_est = np.argmax(y_est_prob,1)

        errors[k] = np.sum(y_est.ravel()!=y_test.ravel(),dtype=float)/y_test.shape[0]
        k+=1

    # Plot the classification error rate
    print('Decision Tree:')
    print('\t error rate: %0.2f%%' % (100 * mean(errors)))
    print('\t top 5 features:')
    pp.pprint(top_features(5, clf.feature_importances_))


# K-Nearest Neighbors
def clf_knn():
    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=2

    # K-fold crossvalidation
    K = 10
    CV = cross_validation.KFold(N, K, shuffle=True)

    errors = np.zeros(K)
    k=0
    for train_index, test_index in CV:
        # print('Crossvalidation fold: {0}/{1}'.format(k + 1, K))

        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]

        clf = KNeighborsClassifier(n_neighbors=K, p=dist);
        clf.fit(X_train, ravel(y_train))
        y_est_prob = clf.predict_proba(X_test)
        y_est = np.argmax(y_est_prob,1)

        errors[k] = np.sum(y_est.ravel()!=y_test.ravel(),dtype=float)/y_test.shape[0]
        k+=1

    # Plot the classification error rate
    print('K-Nearest Neighbors:')
    print('\t error rate: %0.2f%%' % (100 * mean(errors)))


clf_naive_bayes()
clf_dtree()
clf_knn()
