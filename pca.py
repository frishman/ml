
import pandas
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer


class DoPCA:
    def __init__(self, df: pandas.DataFrame, transform: bool, file: str) -> None:
        self.df = df
        self.file = file
        self.loadings = []
        self.variance = []
        self.numpc = int
        self.transform = transform

    def pca(self):
        pca = PCA()
        if self.transform:
            dft = PowerTransformer(method='yeo-johnson', standardize=False).fit_transform(self.df)
            principal_components = pca.fit(dft)
        else:
            principal_components = pca.fit(self.df.to_numpy())
        self.loadings = principal_components.components_
        self.variance = principal_components.explained_variance_ratio_
        self.numpc = principal_components.n_features_

    def print_pca(self):
        pc_list = ["PC" + str(i) for i in list(range(1, self.numpc + 1))]
        print(pc_list)
        loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, self.loadings)))
        print(loadings_df)
        loadings_df['variable'] = self.df.columns.values
        loadings_df = loadings_df.set_index('variable')
        explained = pd.DataFrame.from_dict(dict(zip(pc_list, self.variance.reshape(1, 330).T)))
        explained['variable'] = "% Variance"
        explained = explained.set_index('variable')
        with open(self.file, 'w') as file:
            file.write(explained.append(loadings_df).to_string())

    def get_loadings(self):
        return self.loadings

    def get_variance(self):
        return self.variance



