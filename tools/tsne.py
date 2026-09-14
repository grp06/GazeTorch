import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from tools.utils import palette

def plot_tsne(features, labels, path_save:str, colors=None):
    '''
    features: Nxd feature matrix.
    labels: list of labels (len=N)
    colors: if not None, a collection of colors; if None, I will use a random palette.
        * length >= len(unique(labels))
        * for cases to run this function on multiple data with same label set so they have consistent coloring
    '''
    X = np.array(features)
    Y = np.array(labels)
    labels_unique = np.unique(labels)

    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(X)

    if colors is None:
        n = len(labels_unique)
        colors = palette(n)

    plt.figure()
    # plot w.r.t. colors
    for i, label in enumerate(labels_unique):
        indices = np.where(Y==label)
        x = X_embedded[indices]
        c = colors[i][None,...]
        plt.scatter(x[...,0], x[...,1], c=c, s=2, label=label)

    # plt.legend()
    plt.savefig(path_save)
    print("Voila.")

if __name__=='__main__':
    path_gt = "gt_20p200s.npy"
    D = np.load(f"/data/gazecapture_3d/{path_gt}", allow_pickle=True).item()
    labels = D['subject_cid']
    colors = palette(10)
    
    Xs = np.load("temp_results/0206_subject_corr/features0206_fh1p20.npy")
    plot_tsne(Xs[2000:], labels[2000:], "temp_results/0206_subject_corr/fh1_1.png", colors)

    Xs = np.load("temp_results/0206_subject_corr/features0206_fh2p20.npy")
    plot_tsne(Xs[2000:], labels[2000:], "temp_results/0206_subject_corr/fh2_1.png", colors)