import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
import re
import shap
import matplotlib.pyplot as plt


class Learn:
    def __init__(self, df: pd.DataFrame, dcolumns: list(), tcolumn: str, method: str) -> None:
        self.df = df
        self.dcolumns = dcolumns
        self.tcolumn = tcolumn
        self.method = method
        self.nsplits = 1
        self.split = StratifiedShuffleSplit(n_splits=self.nsplits, test_size=0.2, random_state=42)
        if self.method == "RandomForest":
            self.clf = RandomForestClassifier()
        elif self.method == "SVM":
            self.clf = svm.SVC()
        elif self.method == "NeuralNetwork":
            self.clf = MLPClassifier()
        else:
            print("Unknown classifier: ", method)
            sys.exit()
        self.train_test()

    def get_score(self):
        return self.score

    def metr(self, known: np.array, pred: np.array, original_counts: dict, train_counts: dict,
             train_counts_oversampled: dict):

        labels = np.sort(np.unique(np.concatenate((known, pred))))
        precision, recall, fscore, support = precision_recall_fscore_support(known, pred, labels=labels, zero_division=0)

        confus = confusion_matrix(known, pred)
        print("Accuracy = ", f'{accuracy_score(known, pred):.2f}')
        for i in range(len(labels)):
            wrong = 0
            for j in range(len(confus[i])):
                if j == i:
                    continue
                wrong = wrong + confus[i][j]
            print('%10s' %  labels[i], "\t", original_counts[labels[i]], "\t", train_counts[labels[i]], "\t",
                  train_counts_oversampled[labels[i]], "\t", f'{precision[i]:10.2f}', "\t", f'{recall[i]:10.2f}', "\t",
                  f'{fscore[i]:10.2f}', "\t", f'{support[i]:10.2f}', "\t", '%10i' %  confus[i][i], "\t", '%10i' %  wrong)


    def opt_hyper(self, x_train, y_train):

        if self.method == "RandomForest":
            n_iter = 100
            random_grid = {'n_estimators': [int(x) for x in np.linspace(start=20, stop=2000, num=15)],
                           'max_depth': [int(x) for x in np.linspace(start=1, stop=15, num=10)] + [None],
                           'min_samples_split': [2, 5, 10],
                           'min_samples_leaf': [1, 2, 4],
                           'bootstrap': [True, False]}
        elif self.method == "SVM":
            n_iter = 20
            random_grid = {'C': [0.1, 1, 10, 100, 1000],
                           'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                           'kernel': ['rbf']}
        elif self.method == "NeuralNetwork":
            n_iter = 100
            random_grid = {
                'hidden_layer_sizes': [(10, 30, 10), (20,)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }

        random = RandomizedSearchCV(estimator=self.clf, param_distributions=random_grid, n_iter=n_iter, cv=5, verbose=0,
                                       random_state=42, n_jobs=-1)
        random.fit(x_train, y_train)
        best_grid = random.best_estimator_
        return best_grid

    def run_permutation(self, clf, x_train, y_train):
        feat = dict()
        perm_importance = permutation_importance(clf, x_train, y_train, n_repeats=10,
                                                 random_state=42, n_jobs=-1)
        sorted_importances_idx = perm_importance.importances_mean.argsort()
        test_importances = pd.DataFrame(perm_importance.importances[sorted_importances_idx],
                                        self.dcolumns[sorted_importances_idx]).tail(50)
        for i in range(len(test_importances)):
            p = test_importances.index[i]
            if p not in feat:
                feat[p] = list()
            feat[p].append(test_importances.iloc[i].iat[0])

        print('\n')
        for p in feat.keys():
            if len(feat[p]) == self.nsplits:
                pp = re.sub('[^\\|]*\\|', '', p)
                print("FEATURE " + pp, " ", feat[p])
        print('\n')

    def train_test(self):

        original_counts = self.df[self.tcolumn].value_counts().to_dict()
        for train_index, test_index in self.split.split(self.df, self.df[self.tcolumn]):
            strat_train_set = self.df.iloc[train_index]
            strat_test_set = self.df.iloc[test_index]
            X_train = strat_train_set[self.dcolumns].to_numpy()
            y_train = strat_train_set[self.tcolumn].to_numpy()
            train_counts = strat_train_set[self.tcolumn].value_counts().to_dict()
            oversample = SMOTE()
            X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
            unique, counts = np.unique(y_train_over, return_counts=True)
            train_counts_over = dict(zip(unique, counts))
            X_test = strat_test_set[self.dcolumns].to_numpy()
            y_test = strat_test_set[self.tcolumn].to_numpy()
            best_clf = self.opt_hyper(X_train_over, y_train_over)
            y_pred = best_clf.predict(X_test)
            self.metr(y_test, y_pred, original_counts, train_counts, train_counts_over)


        explainer = shap.TreeExplainer(best_clf)
        shap_values = explainer.shap_values(X_test)
        print(shap_values)
        shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=self.dcolumns)
        # fig = shap.summary_plot(shap_values, train, show=False)
        # plt.savefig('shap.png')









