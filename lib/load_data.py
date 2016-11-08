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

def load_data(standardize):
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

    if standardize:
        X = (X - X.mean()) / (X.max() - X.min())

    return X, y, N, M
