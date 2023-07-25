#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib as mpl
# from msm_pyemma_scripts.plot_helpers import heatmap, annotate_heatmap
import numpy as np
import mdshare
import pyemma
from pyemma import config
from pyemma.util.contexts import settings
import mdtraj as md
import glob
import deeptime
import multiprocessing
import time
import pickle
import heapq
import sys

# sys.path.insert(0, '/scratch365/mfarrugi/HMGR/500ns/analysis/msm-pyemma-scripts/')

config.show_progress_bars = False
# iniital md resolution of 50 ps, or 0.05 ns, ftrzn stride of 4, so 200ps or 0.2ns resolution
ftr_timestep = 0.200
lag_list = [5, 10, 50, 500, 1000, 2000, 2400]
dim_list = [2, 4, 5, 10, 0.95]


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, aspect='auto', **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(len(data[0])), labels=col_labels)
    ax.set_yticks(np.arange(len(data)), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(len(data[0])+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(data)+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, "{:.1f}".format(data[i, j]), **kw)
            texts.append(text)

    return texts


timesteps = ftr_timestep*lag_list


def compare_reductions(prefix:str, pca_model, tica_model, vamp_model, n_dims:int):
    fig, axes = plt.subplots(figsize=(10, 10))
    pca_output = pca_model.get_output()
    pca_concatenated = np.concatenate(pca_output)
    tica_output = tica_model.get_output()
    tica_concatenated = np.concatenate(tica_output)
    vamp_output = vamp_model.get_output()
    vamp_concatenated = np.concatenate(vamp_output)
    pyemma.plots.plot_feature_histograms(np.concatenate([pca_concatenated, tica_concatenated, vamp_concatenated], axis=1), feature_labels=[
                                         ['PCA']*n_dims, ['TICA']*n_dims, ['VAMP']*n_dims], ax=axes[0])
    fig.tight_layout()
    fig.savefig(prefix+'_ftr_hist.png')


def dimensional_analysis(model):
    print()


def lag_analysis(model):
    print()


def plot_IC_densities(prefix:str, model, num_ICs:int):
    model_output = model.get_output(num_ICs, num_ICs, figsize=(10,10))
    concatenated = np.concatenate(model_output)
    fig, axes = plt.subplots
    for i in range(num_ICs):
        for j in range(num_ICs):
            pyemma.plots.plot_density(concatenated[:,i], concatenated[:,j], ax=axes[i,j], cbar=True, alpha=0.1)
    fig.tight_layout()
    fig.savefig(prefix+'_IC_densities.png')


