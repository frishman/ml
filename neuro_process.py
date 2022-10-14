import numpy as np
import pandas as pd
from pca import DoPCA
from plot import scatter_dict
from libr import impscale
from libr import dict_append
from learn import Learn


def preprocess_nd(df):

    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(' ', '_', regex=True)
        df[c] = df[c].astype(str).str.replace('\'', '_', regex=True)
        df[c] = df[c].astype(str).str.replace('(', '_', regex=True)
        df[c] = df[c].astype(str).str.replace(')', '_', regex=True)
        df[c] = df[c].astype(str).str.replace(',', '_', regex=True)
        df[c] = df[c].astype(str).str.replace('\\/', '_', regex=True)

    if 'Age' in df.columns:
        df["Age"].replace({"90+": "90"}, inplace=True)
        df["Age"] = df["Age"].astype('float64')

    return df


def run_learn_individual(prot_file: str, prot_index: str, traits_file: str, traits_index:str, group: str):

    clinical_data = pd.read_csv(traits_file, index_col=traits_index)
    clinical_data = preprocess_nd(clinical_data)
    clinical_data = clinical_data.dropna(how='any')

    proteomics_data = pd.read_csv(prot_file, index_col=prot_index)
    proteomics_data = proteomics_data.dropna(how='all', axis=1)
    proteomics_data = proteomics_data.dropna(how='all')
    if proteomics_data.min().min() > 0:
        proteomics_data = (np.log2(proteomics_data))

    # boxframe(nd_proteomics_data, 3, "/Users/frishman/Downloads/bx_sample.pdf")

    columns = list(set(proteomics_data.columns.tolist()) & set(clinical_data.index.tolist()))
    proteomics_data = proteomics_data[columns]

    proteomics_data = impscale(proteomics_data)
    proteomics_data = proteomics_data.T
    prot_columns = proteomics_data.columns
    joint = proteomics_data.join(clinical_data)

    proteomics_data = proteomics_data.join(clinical_data[group])

    min_group_size = 15
    for g in proteomics_data[group].unique():
        if len(proteomics_data[proteomics_data[group] == g]) < min_group_size:
            proteomics_data.drop(proteomics_data.index[proteomics_data[group] == g], inplace=True)

    # enc = OrdinalEncoder()
    # nd_proteomics_data[["Group"]] = enc.fit_transform(nd_proteomics_data[["Group"]])

    # boxframe(nd_proteomics_data, 5, "/Users/frishman/Downloads/bx_gene.pdf")
    X = proteomics_data[prot_columns]
    y = proteomics_data[group]
    Learn(X, y, "RandomForest")

def run_learn_combined(h5ad_file):
    adata = sc.read_h5ad(h5ad_file)
    X = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
    y = pd.DataFrame(adata.obs['diagnosis']).squeeze()
    Learn(X, y, "RandomForest")


def run_pca(prot: pd.DataFrame, clin: pd.DataFrame, pcatxt: str, pcapdf: str):

    p = DoPCA(pd, False, pcatxt)
    p.pca()
    loadings1 = p.get_loadings()[0]
    loadings2 = p.get_loadings()[1]

    ld1 = dict()
    ld2 = dict()
    for i in range(len(clin.columns.tolist())):
        grp = clin.iloc[i]['Group'][:7]
        dict_append(ld1, grp, loadings1[i])
        dict_append(ld2, grp, loadings2[i])

    scatter_dict(ld1, ld2, "PC1", "PC2", pcapdf)



