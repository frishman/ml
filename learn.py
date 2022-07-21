import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
import imblearn
from imblearn.under_sampling import RandomUnderSampler
import sys
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import shap


class Learn:
    def __init__(self, df: pd.DataFrame, dcolumns: list(), tcolumn: str, method: str) -> None:
        self.df = df
        self.dcolumns = dcolumns
        self.tcolumn = tcolumn
        self.split = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
        if method == "RandomForest":
            self.clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=5, n_jobs=-1)
        elif method == "SVM":
            self.clf = svm.SVC(kernel='linear')
        elif method == "NeuralNetwork":
            self.clf = MLPClassifier(
            #self.clf = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
                hidden_layer_sizes=(50,),
                max_iter=15,
                alpha=1e-4,
                solver="sgd",
                verbose=False,
                random_state=1,
                learning_rate_init=0.1
            )
        else:
            print("Unknown classifier: ", method)
            sys.exit()
        self.score = self.train_test()

    def get_score(self):
        return self.score

    def metr(self, known: np.array, pred: np.array):

        labels = np.sort(np.unique(np.concatenate((known, pred))))
        precision, recall, fscore, support = precision_recall_fscore_support(known, pred, labels=labels)

        for i in range(len(labels)):
            print(labels[i], " ", f'{precision[i]:.2f}', " ", f'{recall[i]:.2f}', " ", f'{fscore[i]:.2f}', " ",
                  f'{support[i]:.2f}')
        print("Accuracy = ", f'{accuracy_score(known, pred):.2f}'   )
        print("Balanced accuracy score = ", f'{balanced_accuracy_score(known, pred):.2f}')
        print("Confusion matrix")
        print(confusion_matrix(known, pred), '\n')

    def opt_hyper_rf(self, x_train, y_train):

        n_estimators = [int(x) for x in np.linspace(start=20, stop=2000, num=15)]
        min_samples_leaf = [1, 2, 4]
        min_samples_split = [2, 5, 10]
        max_depth = [int(x) for x in np.linspace(start=1, stop=15, num=10)]
        max_depth.append(None)
        bootstrap = [True, False]

        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf_random = RandomizedSearchCV(estimator=self.clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=1,
                                       random_state=42, n_jobs=-1)
        rf_random.fit(x_train, y_train)
        #print(rf_random.best_params_)
        best_grid = rf_random.best_estimator_
        return best_grid

    def train_test(self):

        rus = RandomUnderSampler(sampling_strategy='majority')

        print("Label counts in the original dataset:")
        print(self.df[self.tcolumn].value_counts().to_string(), '\n')

        count = 1
        feat = dict()
        feat_best = dict()
        for train_index, test_index in self.split.split(self.df, self.df[self.tcolumn]):
            print("Split # ", count)
            strat_train_set = self.df.iloc[train_index]
            strat_test_set = self.df.iloc[test_index]
            X_train = strat_train_set[self.dcolumns].to_numpy()
            y_train = strat_train_set[self.tcolumn].to_numpy()
            print("Label counts in the traning set before oversampling:")
            print(strat_train_set[self.tcolumn].value_counts().to_string(), '\n')
            oversample = SMOTE()
            X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
#            X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
            print("Label counts in the training set after oversampling:")
            print(pd.DataFrame(y_train_over).value_counts().to_string(), '\n')
            X_test = strat_test_set[self.dcolumns].to_numpy()
            y_test = strat_test_set[self.tcolumn].to_numpy()
            best_clf = self.opt_hyper_rf(X_train_over, y_train_over)
            self.clf.fit(X_train_over, y_train_over)

            y_pred = self.clf.predict(X_test)
            y_pred_best = best_clf.predict(X_test)
            print("Normal")
            self.metr(y_test, y_pred)
            importances = self.clf.feature_importances_

            imp = pd.DataFrame({'col_name': importances}, self.dcolumns).sort_values(by='col_name', ascending=False).head(50)
            for i in range(len(imp.index)):
                p = imp.index[i]
                if p not in feat:
                    feat[p] = list()
                feat[p].append(imp.iloc[i]['col_name'])
            count += 1

            print("Best")
            self.metr(y_test, y_pred_best)
            importances = best_clf.feature_importances_

            imp = pd.DataFrame({'col_name': importances}, self.dcolumns).sort_values(by='col_name', ascending=False).head(50)
            for i in range(len(imp.index)):
                p = imp.index[i]
                if p not in feat_best:
                    feat_best[p] = list()
                feat_best[p].append(imp.iloc[i]['col_name'])
            count += 1

        print("Normal")
        print(len(feat.keys()))
        for p in feat.keys():
            print(p, ": ", feat[p])

        print("Best")
        print(len(feat_best.keys()))
        for p in feat_best.keys():
            print(p, ": ", feat_best[p])

        # explainer = shap.TreeExplainer(self.clf)
        # shap_values = explainer.shap_values(X_test)
        # print(shap_values)
        # shap.summary_plot(shap_values, X_test, plot_type="bar")









