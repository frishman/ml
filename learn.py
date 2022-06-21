import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import sys



class Learn:
    def __init__(self, df: pd.DataFrame, dcolumns: list(), tcolumn: str, method: str) -> None:
        self.df = df
        self.dcolumns = dcolumns
        self.tcolumn = tcolumn
        self.split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        if method == "RandomForest":
            self.clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        elif method == "SVM":
            self.clf = svm.SVC(kernel='linear')
        elif method == "NeuralNetwork":
            self.clf = MLPClassifier(
                hidden_layer_sizes=(50,),
                max_iter=15,
                alpha=1e-4,
                solver="sgd",
                verbose=False,
                random_state=1,
                learning_rate_init=0.1
            )

            #self.clf = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
        else:
            print("Unknown classifier: ", method)
            sys.exit()
        self.score = self.train_test()

    def train_test(self):

        score = cross_val_score(self.clf, self.df[self.dcolumns].to_numpy(), self.df[self.tcolumn].to_numpy(),
                        cv=10)
        return score.mean()
    # print(cross_val_score(self.clf, self.df[self.dcolumns].to_numpy(), self.df[self.tcolumn].to_numpy(),
    #                 cv=10, scoring="accuracy"))
    # print(cross_val_score(self.clf, self.df[self.dcolumns].to_numpy(), self.df[self.tcolumn].to_numpy(),
    #                 cv=10, scoring="accuracy"))
    # print(cross_val_score(self.clf, self.df[self.dcolumns].to_numpy(), self.df[self.tcolumn].to_numpy(),
    #                 cv=10, scoring="accuracy"))

    def get_score(self):
        return self.score

    def train_test_alt(self):
        count = 0

        for train_index, test_index in self.split.split(self.df, self.df[self.tcolumn]):
            strat_train_set = self.df.iloc[train_index]
            strat_test_set = self.df.iloc[test_index]
            X_train = strat_train_set[self.dcolumns].to_numpy()
            y_train = strat_train_set[self.tcolumn].to_numpy()
            X_test = strat_test_set[self.dcolumns].to_numpy()
            y_test = strat_test_set[self.tcolumn].to_numpy()

            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)

            print("Split =", count)
            print(strat_test_set)
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            print(accuracy_score(y_test, y_pred))

            count = count + 1


# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
#
# np.random.seed()
# rng = np.random.default_rng()
# data = rng.uniform(-1, 1, (30, 6))
#
# df = pd.DataFrame(data)
# df.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
# nclass = int(len(df)/2)
# df['class'] = [0] * nclass + [1] * nclass
#
# #train_set, test_set = split_train_test(df, 0.2)
# train_set1, test_set1 = train_test_split(df, test_size=0.2, random_state=42)
# train_set2, test_set2 = train_test_split(df, test_size=0.2, random_state=12)




