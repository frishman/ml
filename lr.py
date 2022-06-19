from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import datasets
import sys

X, y = load_iris(return_X_y=True)

iris = load_iris()
data = iris.data
target = iris.target
names = iris.target_names
feature = iris.feature_names

df = pd.DataFrame(data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].replace(to_replace= [0, 1, 2], value = ['setosa', 'versicolor', 'virginica'])

print(df.sample(frac=0.1))

cls = LogisticRegression(C=100000.0)
cls.fit(X, y)
X_pred = [5.1, 3.2, 1.5, 0.5]
y_pred = cls.predict([X_pred])
print(y_pred)

sys.exit()

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target
logreg = LogisticRegression(C=1e5)
logreg.fit(X, Y)
X_pred = [5.1, 3.2, 1.5, 0.5]
y_pred = logreg.predict([X_pred])
print(y_pred)



iris = load_iris()

# Create a dataframe
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target

# Let's see a sample of created df
df.sample(frac=0.01)

# Let's see target names
targets = iris.target_names
print(targets)
['setosa' 'versicolor' 'virginica']
# Prepare training data for building the model
X_train = df.drop(['target'], axis=1)
y_train = df['target']

# Instantiate the model
cls = LogisticRegression()

# Train/Fit the model
cls.fit(X_train, y_train)

# Make prediction using the model
X_pred = [5.1, 3.2, 1.5, 0.5]
y_pred = cls.predict([X_pred])

print("Prediction is: {}".format(targets[y_pred]))

print("Hello")
