import pandas as pd
from pca import DoPCA
from plot import plot_pca


def prepplot(replicate: str, marker: str, title: str):
    x = list()
    y = list()
    col = list()
    name = list()
    for i in range(0, len(columns)):
        if replicate in columns[i]:
            x.append(p.get_loadings()[0][i])
            y.append(p.get_loadings()[1][i])
            if columns[i].startswith("M"):
                col.append("red")
                name.append("Mock")
            elif columns[i].startswith("P"):
                col.append("green")
                name.append("Pr8")
            elif columns[i].startswith("SD"):
                col.append("blue")
                name.append("SC35MdNS1")
            elif columns[i].startswith("SW"):
                col.append("magenta")
                name.append("SC35M")
    return x, y, col, name, marker, title


columns = ['MS2T1', 'MS3T1', 'PS1T1', 'PS2T1', 'PS3T1', 'SDS1T1', 'SDS2T1', 'SDS3T1', 'SWS1T1', 'SWS2T1', 'SWS3T1']

# df = pd.read_csv("/Users/frishman/Bioinformatics/projects/HERV/raw_countTable.tsv", sep='\t').loc[:, columns]

df = pd.read_csv("/Users/frishman/Bioinformatics/projects/HERV/vsd_count_pca.tsv", sep='\t').loc[:, columns]

p = DoPCA(df, False, "/Users/frishman/Bioinformatics/projects/HERV/pca.txt")
p.pca()
p.print_pca()
var = p.get_variance()
var1 = "{:.0f}".format(100 * var[0])
var2 = "{:.0f}".format(100 * var[1])

graph = list()
graph.append(prepplot("S1", "s", "Sample 1"))
graph.append(prepplot("S2", "o", "Sample 2"))
graph.append(prepplot("S3", "+", "Sample 3"))
plot_pca(graph, "PC1 (" + var1 + "%)", "PC2 (" + var2 + "%)", "Biological replicate", "Cell type",
         "/Users/frishman/Bioinformatics/projects/HERV/plot.png")