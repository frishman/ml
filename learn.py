import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
import re
import shap
import libr
import warnings
from sklearn.exceptions import ConvergenceWarning

class Learn:
    def __init__(self, X, y, method) -> None:
        self.method = method
        self.nsplits = 1
        self.split = StratifiedShuffleSplit(n_splits=self.nsplits, test_size=0.2, random_state=42)
        self.set_method()
        self.feature_names = list(X.columns)
        self.train_test(X, y)

    def set_method(self):
        if self.method == "RandomForest":
            self.clf = RandomForestClassifier()
        elif self.method == "SVM":
            self.clf = svm.SVC(probability=True)
        elif self.method == "NeuralNetwork":
            self.clf = MLPClassifier(verbose=False, max_iter=200)
        else:
            supported_methods = ["RandomForest", "SVM", "NeuralNetwork"]
            raise ValueError(f"'method' should be one of following: {supported_methods}")

    def metr(self, known: np.array, pred: np.array, original_counts: dict, train_counts: dict,
             train_counts_oversampled: dict):

        labels = np.sort(np.unique(np.concatenate((known, pred))))
        precision, recall, fscore, support = precision_recall_fscore_support(known, pred, labels=labels, zero_division=0)

        confus = confusion_matrix(known, pred)
        print("Accuracy = ", f'{accuracy_score(known, pred):.2f}')
        print(f'{"Label":>10s}', "\t", f'{"# Orig":>10s}', "\t", f'{"# Train":>10s}',"\t",
              f'{"# Train oversmpl":>15s}', "\t", f'{"Precision":>10s}', "\t", f'{"Recall":>10s}',"\t",
              f'{"F score":>10s}', "\t", f'{"Support":>10s}', "\t", f'{"Correct":>10s}', "\t", f'{"Wrong":>10s}')
        for i in range(len(labels)):
            wrong = 0
            for j in range(len(confus[i])):
                if j == i:
                    continue
                wrong = wrong + confus[i][j]
            print(f'{labels[i]:>10}', "\t", f'{original_counts[labels[i]]:>10}', "\t", f'{train_counts[labels[i]]:>10}', "\t",
                  f'{train_counts_oversampled[labels[i]]:>15}', "\t", f'{precision[i]:>10.2f}', "\t", f'{recall[i]:>10.2f}', "\t",
                  f'{fscore[i]:>10.2f}', "\t", f'{support[i]:>10d}', "\t", f'{confus[i][i]:>10d}', "\t", f'{wrong:>10d}')


    def opt_hyper(self, x_train, y_train):

        if self.method == "RandomForest":
            n_iter = 10
            random_grid = {'n_estimators': [int(x) for x in np.linspace(start=20, stop=2000, num=15)],
                           'max_depth': [int(x) for x in np.linspace(start=1, stop=15, num=10)] + [None],
                           'min_samples_split': [2, 5, 10],
                           'min_samples_leaf': [1, 2, 4],
                           'bootstrap': [True, False]}

        elif self.method == "SVM":
            n_iter = 100
            random_grid = {'C': [0.1, 1, 10, 100, 1000],
                           'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                           'kernel': ['rbf']}
        elif self.method == "NeuralNetwork":
            n_iter = 30
            random_grid = {
                'hidden_layer_sizes': [(10, 30, 10), (20,)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }

        random = RandomizedSearchCV(estimator=self.clf, param_distributions=random_grid, n_iter=n_iter, cv=5, verbose=0,
                                       random_state=42)
        random.fit(x_train, y_train)
        best_grid = random.best_estimator_
        return best_grid

    # def run_permutation(self, clf, x_train, y_train):
    #     feat = dict()
    #     perm_importance = permutation_importance(clf, x_train, y_train, n_repeats=10,
    #                                              random_state=42, n_jobs=-1)
    #     sorted_importances_idx = perm_importance.importances_mean.argsort()
    #     test_importances = pd.DataFrame(perm_importance.importances[sorted_importances_idx],
    #                                     self.dcolumns[sorted_importances_idx]).tail(50)
    #     for i in range(len(test_importances)):
    #         p = test_importances.index[i]
    #         if p not in feat:
    #             feat[p] = list()
    #         feat[p].append(test_importances.iloc[i].iat[0])
    #
    #     print('\n')
    #     for p in feat.keys():
    #         if len(feat[p]) == self.nsplits:
    #             pp = re.sub('[^\\|]*\\|', '', p)
    #             print("FEATURE " + pp, " ", feat[p])
    #     print('\n')

    def explain(self, clf, x):

        if self.method == "RandomForest":
            explainer = shap.TreeExplainer(clf, x)
        elif self.method == "SVM" or self.method == "NeuralNetwork":
            explainer = shap.KernelExplainer(clf.predict_proba, x)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            shap_values = explainer.shap_values(x, check_additivity=False)
        importances = dict()
        feature_names = list(x.columns)
        for s in shap_values:
            for i in range(s.shape[1]):
                if feature_names[i] not in importances:
                    importances[feature_names[i]] = np.mean(np.abs(s[:, i]))
                else:
                    importances[feature_names[i]] = importances[feature_names[i]] + np.mean(np.abs(s[:, i]))
#        shap.summary_plot(shap_values, x, plot_type="bar", feature_names=self.dcolumns)
        return importances

    def train_test(self, X, y):

        importances_splits = dict()
        original_counts = libr.array_counts(y)
        for train_index, test_index in self.split.split(X, y):
            X_train, X_test = X.iloc[train_index].to_numpy(), X.iloc[test_index].to_numpy()
            y_train, y_test = y.iloc[train_index].to_numpy(), y.iloc[test_index].to_numpy()
            oversample = SMOTE()
            X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
            train_counts = libr.array_counts(y_train)
            train_counts_over = libr.array_counts(y_train_over)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                best_clf = self.opt_hyper(X_train_over, y_train_over)
                y_pred = best_clf.predict(X_test)
            self.metr(y_test, y_pred, original_counts, train_counts, train_counts_over)
            importances = self.explain(best_clf, X.iloc[test_index])
            for p in importances.keys():
                if p not in importances_splits.keys():
                    importances_splits[p] = importances[p]
                else:
                    importances_splits[p] = importances_splits[p] + importances[p]

        importances_values = np.array(list(importances_splits.values()))
        importances_idx = np.argsort(importances_values)
        #sorted_importance_values = importances_values[importances_idx][-20:]
        #print(sorted_importance_values)
        protein_ids = np.array(self.feature_names)[importances_idx][-20:]
        print()
        for p in protein_ids:
            print(p)
            if type(p) is not str:
                p = str(p)
            p = re.sub('[^\\|]*\\|', '', p)
            print("FEATURE " + p)
        print()
