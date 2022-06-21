import sys

import numpy as np
import pandas as pd
from neuro_preprocess import preprocess_nd
from pca import DoPCA
from plot import scatter_dict
from libr import impscale
from libr import dict_append
from sklearn.preprocessing import OrdinalEncoder
from learn import Learn
from sklearn import datasets


traits_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/UPenn_Multiple_Neurodegenerative_Diseases/Discovery_LFQ_Proteomics/data/0.Traits.csv"
proteomics_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/UPenn_Multiple_Neurodegenerative_Diseases/Discovery_LFQ_Proteomics/data/2.unregressed_batch-corrected_LFQ_intensity.csv"
joint_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/UPenn_Multiple_Neurodegenerative_Diseases/Discovery_LFQ_Proteomics/data/joint.xlsx"

nd_clinical_data = pd.read_csv(traits_file, index_col='MaxQuant ID')
nd_clinical_data = preprocess_nd(nd_clinical_data)
groups= nd_clinical_data['Group'].unique()

nd_proteomics_data = pd.read_csv(proteomics_file, index_col='Unnamed: 0')
nd_proteomics_data = (np.log2(nd_proteomics_data))


#boxframe(nd_proteomics_data, 3, "/Users/frishman/Downloads/bx_sample.pdf")

nd_proteomics_data_all = nd_proteomics_data
columns = list(set(nd_proteomics_data.columns.tolist()) & set(nd_clinical_data.index.tolist()))
nd_proteomics_data = nd_proteomics_data[columns]

nd_proteomics_data = impscale(nd_proteomics_data)

p = DoPCA(nd_proteomics_data, False, "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/pca.txt")
p.pca()
loadings1 = p.get_loadings()[0]
loadings2 = p.get_loadings()[1]

ld1 = dict()
ld2 = dict()
for i in range(len(nd_proteomics_data.columns.tolist())):
    grp = nd_clinical_data.iloc[i]['Group'][:7]
    dict_append(ld1, grp, loadings1[i])
    dict_append(ld2, grp, loadings2[i])

scatter_dict(ld1, ld2, "PC1", "PC2", "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/pca.pdf")
nd_proteomics_data = nd_proteomics_data.T
prot_columns = nd_proteomics_data.columns
nd_proteomics_data = nd_proteomics_data.join(nd_clinical_data['Group'])

min_group_size = 15
for g in nd_proteomics_data["Group"].unique():
    if len(nd_proteomics_data[nd_proteomics_data["Group"] == g]) < min_group_size:
        nd_proteomics_data.drop(nd_proteomics_data.index[nd_proteomics_data['Group'] == g], inplace=True)

enc = OrdinalEncoder()
nd_proteomics_data[["Group"]] = enc.fit_transform(nd_proteomics_data[["Group"]])



#boxframe(nd_proteomics_data, 5, "/Users/frishman/Downloads/bx_gene.pdf")

# ln = Learn(nd_proteomics_data, prot_columns, "Group", "SVM")
# print("SMV: ", ln.get_score())
#
# ln = Learn(nd_proteomics_data, prot_columns, "Group", "RandomForest")
# print("Random Forest: ", ln.get_score())
#
# ln = Learn(nd_proteomics_data, prot_columns, "Group", "NeuralNetwork")
# print("Neural Network: ", ln.get_score())

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
columns = df.columns
df["target"] = iris.target
ln = Learn(df, columns, "target", "SVM")
print("SMV: ", ln.get_score())
ln = Learn(df, columns, "target", "RandomForest")
print("Random Forest: ", ln.get_score())
ln = Learn(df, columns, "target", "NeuralNetwork")
print("Neural network: ", ln.get_score())









