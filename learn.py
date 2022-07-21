import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
import re
#import shap


class Learn:
    def __init__(self, df: pd.DataFrame, dcolumns: list(), tcolumn: str, method: str) -> None:
        self.df = df
        self.dcolumns = dcolumns
        self.tcolumn = tcolumn
        self.nsplits = 5
        self.split = StratifiedShuffleSplit(n_splits=self.nsplits, test_size=0.2, random_state=42)
        if method == "RandomForest":
            self.clf = RandomForestClassifier(n_jobs=-1)
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

        rf_random = RandomizedSearchCV(estimator=self.clf, param_distributions=random_grid, n_iter=100, cv=5, verbose=0,
                                       random_state=42, n_jobs=-1)
        rf_random.fit(x_train, y_train)
        #print(rf_random.best_params_)
        best_grid = rf_random.best_estimator_
        return best_grid

    def train_test(self):

        original_counts = self.df[self.tcolumn].value_counts().to_dict()
        feat = dict()
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
            best_clf = self.opt_hyper_rf(X_train_over, y_train_over)
            y_pred = best_clf.predict(X_test)
            self.metr(y_test, y_pred, original_counts, train_counts, train_counts_over)
            imp = pd.DataFrame({'col_name': best_clf.feature_importances_}, self.dcolumns).sort_values(by='col_name', ascending=False).head(50)
            for i in range(len(imp.index)):
                p = imp.index[i]
                if p not in feat:
                    feat[p] = list()
                feat[p].append(imp.iloc[i]['col_name'])

        print()
        for p in feat.keys():
            if len(feat[p]) == self.nsplits:
                p = re.sub('[^\\|]*\\|', '', p)
                print("FEATURE " + p)
        print('\n\n')



        # explainer = shap.TreeExplainer(self.clf)
        # shap_values = explainer.shap_values(X_test)
        # print(shap_values)
        # shap.summary_plot(shap_values, X_test, plot_type="bar")









