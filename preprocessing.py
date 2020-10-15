import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def QC_control_plot(anndata):
    # Thresholding decision: counts

    p1 = sb.distplot(anndata.obs['n_counts'], kde=False)
    plt.show()

    p2 = sb.distplot(anndata.obs['n_counts'][anndata.obs['n_counts'] < 3000], kde=False, bins=60)
    plt.show()

    p3 = sb.distplot(anndata.obs['n_counts'][anndata.obs['n_counts'] > 20000], kde=False, bins=60)
    plt.show()

    # Thresholding decision: genes
    p4 = sb.distplot(anndata.obs['n_genes'], kde=False, bins=60)
    plt.show()

    p5 = sb.distplot(anndata.obs['n_genes'][anndata.obs['n_genes'] < 1500], kde=False, bins=60)
    plt.show()

