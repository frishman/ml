import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
print(ad.__version__)

counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
ar = np.random.poisson(1, size=(100, 2000))

adata = ad.AnnData(ar)

print(adata.n_obs)
print(adata.obs_names)
print(adata.n_vars)
print(adata.var_names)

adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]


print(adata.n_obs)
print(adata.obs_names)
print(adata.n_vars)
print(adata.var_names)

i = 5
print(f"""First line {i:d}
second line
third line""")
