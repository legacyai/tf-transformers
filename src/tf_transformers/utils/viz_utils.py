import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def compute_scores(vectors):
    corr = np.inner(vectors, vectors)
    cmax = np.max(corr)
    corr /= cmax
    return corr


def plot_similarity(labels, features1, features2, rotation, title1="Model1", title2="Model2"):

    corr1 = compute_scores(features1)
    corr2 = compute_scores(features2)
    sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
    sns.set_context("poster")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
    fig.subplots_adjust(wspace=0.02)

    sns.set(font_scale=1.0)
    g1 = sns.heatmap(
        corr1,
        ax=ax1,
        cbar=False,
        yticklabels=labels,
        xticklabels=labels,
        vmin=np.min(corr1),
        vmax=np.max(corr1),
        cmap="Blues",
    )

    g2 = sns.heatmap(
        corr2,
        ax=ax2,
        cbar=False,
        xticklabels=labels,
        vmin=np.min(corr2),
        vmax=np.max(corr2),
        cmap="Blues",
    )
    g2.set(yticks=[])
    fig.colorbar(ax2.collections[0], ax=ax1, location="right", use_gridspec=False, pad=0.01)
    fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.01)

    g1.set_title(title1)
    g2.set_title(title2)
