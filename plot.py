import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.backends.backend_pdf

def plot_pca(graph: list, xlab: str, ylab: str, first_title: str, second_title: str, file: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    lgd = list()
    for i in range(len(graph)):
        ax.scatter(graph[i][0], graph[i][1], c=graph[i][2], marker=graph[i][4])
        lgd.append(graph[i][5])

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    legend = ax.legend(lgd, title=first_title,
                       bbox_to_anchor=(1.22, 0.5), ncol=1)
    legend.legendHandles[0].set_color('black')
    legend.legendHandles[1].set_color('black')
    legend.legendHandles[2].set_color('black')

    ax.add_artist(legend)
    patches = list()
    for i in range(len(graph[1][2])):
        patches.append(mpatches.Patch(color=graph[1][2][i], label=graph[1][3][i]))
    ax.legend(handles=patches, title=second_title, bbox_to_anchor=(1.2, 1), ncol=1)

    plt.subplots_adjust(right=0.85)
    plt.savefig(file)


def group_hist(df: pd.DataFrame, var: str, group_column: str, outfile: str):

    groups = np.unique(df[group_column].values.tolist())
    groups = groups[groups != 'nan']
    gr_count = 0

    cmap = matplotlib.cm.get_cmap('tab20')

    break_out_flag = False
    ncolumns :int = 3
    nrows = int(len(groups) / ncolumns)
    if (len(groups) % ncolumns) != 0:
        nrows += 1
    fig, ax = plt.subplots(nrows, ncolumns, figsize=(15, 15), sharey=True)
    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.6)
    min_var = df[var].min()
    max_var = df[var].max()

    for i in range(nrows):
        for j in range(ncolumns):
            gr = groups[gr_count]
            group_data = df[df[group_column] == gr]
            plt.title(gr)
            ax[i][j].set_xlim([min_var, max_var])
            ax[i][j].set_title(gr[:35])
            sns.histplot(ax=ax[i][j], data=group_data, x=var, color=cmap(gr_count), label=gr, legend=True)
            gr_count += 1
            if gr_count == len(groups) and j < ncolumns-1:
                for jj in [j + 1, ncolumns - 1]:
                    ax[i][jj].set_visible(False)
                    ax[i][jj].set_visible(False)
                break_out_flag = True
                break
        if break_out_flag:
            break

    if os.path.isfile(outfile):
        os.remove(outfile)
    plt.savefig(outfile, format="pdf")


def boxframe(df, nplots, outfile):
    if os.path.isfile(outfile):
        os.remove(outfile)

    pdf = matplotlib.backends.backend_pdf.PdfPages(outfile)

    nplots = 3
    ngraphs = int(len(df.columns) / nplots)
    if len(df.columns) % nplots != 0:
        ngraphs += 1

    for g in range(ngraphs):
        colnames = df.columns[g * nplots:g * nplots + nplots].tolist()
        df2 = pd.DataFrame(data=df, columns=colnames)
        fig, ax = plt.subplots(1, 1)
        ax = sns.boxplot(data=df2)
        pdf.savefig(fig)

    pdf.close()

def scatter_dict(x: dict(), y: dict(), xlab: str, ylab: str, outfile: str):

    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('tab20')

    color_count = 0
    for k in x.keys():
        ax.scatter(x[k], y[k], color=cmap.colors[color_count], label=k)
        color_count += 1
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    ax.legend()
    if os.path.isfile(outfile):
        os.remove(outfile)
    plt.savefig(outfile, format="pdf")
