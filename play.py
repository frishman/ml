import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


known = np.array(['a', 'b', 'a', 'a', 'a', 'b', 'b', 'a', 'b', 'a', 'c', 'c', 'c', 'a', 'c', 'c', 'c', 'c', 'c', 'c'])
pred  = np.array(['a', 'b', 'b', 'a', 'a', 'b', 'b', 'a', 'b', 'a', 'c', 'c', 'c', 'c', 'c', 'b', 'c', 'c', 'c', 'c'])
labels = np.sort(np.unique(np.concatenate((known, pred))))

print(classification_report(known, pred))
precision, recall, fscore, support = precision_recall_fscore_support(known, pred, labels=labels)

for i in range(len(labels)):
    print(labels[i], " ", f'{precision[i]:.2f}', " ", f'{recall[i]:.2f}', " ", f'{fscore[i]:.2f}', " ", f'{support[i]:.2f}')
print(accuracy_score(known,pred))
print(known)
print(pred)
print(confusion_matrix(known, pred))
