import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

def impscale(df: pd.DataFrame):

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()), ])
    X = num_pipeline.fit_transform(df)
    df = pd.DataFrame(X, columns=df.columns,
                                  index=df.index)
    return df

def dict_append(dct, key, val):

    if key not in dct:
        dct[key] = list()
    dct[key].append(val)
