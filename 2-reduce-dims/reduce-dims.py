#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib as mpl
#from msm_pyemma_scripts.plot_helpers import heatmap, annotate_heatmap

import mdshare
import numpy as np
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

#sys.path.insert(0, '/scratch365/mfarrugi/HMGR/500ns/analysis/msm-pyemma-scripts/')

config.show_progress_bars = False
ftr_timestep = 0.200 # iniital md resolution of 50 ps, or 0.05 ns, ftrzn stride of 4, so 200ps or 0.2ns resolution
lag_list = [5, 10, 50, 500, 1000, 2000, 2400]
dim_list = [2, 4, 10, 0.95]

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

def mode_ftr_analysis(ftr_corr, prefix:str, data_labels, cutoff0 = 0.9, cutoff=0.75, topN=None, data_label='data'):
    
    num_modes = len(ftr_corr[0])
    
    fig,ax = plt.subplots(figsize=[10,10]) 
    
    top_indices = []
    top_vals = []
    if topN is not None:
        for m in range(num_modes):
            top_m = list(map(list, zip(*heapq.nlargest(topN, enumerate(ftr_corr[:,m]), key=lambda x: x[1]))))
            top_indices.append(top_m[0])
            top_vals.append(top_m[1])
    else:
        raise Exception(NotImplementedError())
        top_pca_indices_1 = np.vstack((np.argwhere(ftr_corr[:,0] > 0.95), np.argwhere(pc_ftr_corr[:,0] < -0.95)))
        print("top pca indices len "+str(len(top_pca_indices_1))+"\n"+str(top_pca_indices_1))
        top_pca_vals_1 = pc_ftr_corr[top_pca_indices_1]

        top_pca_indices_2 = np.vstack((np.argwhere(pc_ftr_corr[:,1] > 0.8), np.argwhere(pc_ftr_corr[:,1] < -0.8)))
        print("top pca indices len "+str(len(top_pca_indices_2))+"\n"+str(top_pca_indices_2))
        top_pca_vals_2 = pc_ftr_corr[top_pca_indices_2]
        #top_indices.append()

    unique_indices = np.unique(top_indices)
    unique_vals = [ftr_corr[i,:] for i in unique_indices]
    unique_labels = [data_labels[i] for i in unique_indices]


    im, cbar = heatmap(np.array(unique_vals), unique_labels, range(num_modes), ax=ax, cbarlabel="contribution")
    texts = annotate_heatmap(im, np.array(unique_vals), valfmt="{x:.1f}")
#    plt.clim(-1,1)
    fig.tight_layout()
    plt.show()
    #plt.clim(-1,1)
    cbar = fig.colorbar(im, ax=ax) # added
    cbar.set_ticks([-1.0,0.0, 1.0])
    cbar.draw_all()
    ax.imshow(top_vals, cmap='bwr', vmin=-1, vmax=1, aspect='auto')


    fig.savefig(prefix+'_modes.png')


    fig.clear()
    plt.cla()
    plt.clf()
    print("done mode analysis")

def vamp_sv_analysis(left_singular_vectors, prefix:str, data_labels, topN=None, data_label='data'):
    
    num_modes = len(left_singular_vectors[0])
    
    fig,ax = plt.subplots()
    
    top_indices = []
    top_vals = []
    if topN is not None:
        top_vectors = left_singular_vectors[range(topN)]
        for m in range(num_modes):
            top_m = list(map(list, zip(*heapq.nlargest(topN, enumerate(left_singular_vectors[:,m]), key=lambda x: x[1]))))
            top_indices.append(top_m[0])
            top_vals.append(top_m[1])
    else:
        raise Exception(NotImplementedError())
        top_pca_indices_1 = np.vstack((np.argwhere(ftr_corr[:,0] > 0.95), np.argwhere(pc_ftr_corr[:,0] < -0.95)))
        print("top pca indices len "+str(len(top_pca_indices_1))+"\n"+str(top_pca_indices_1))
        top_pca_vals_1 = pc_ftr_corr[top_pca_indices_1]

        top_pca_indices_2 = np.vstack((np.argwhere(pc_ftr_corr[:,1] > 0.8), np.argwhere(pc_ftr_corr[:,1] < -0.8)))
        print("top pca indices len "+str(len(top_pca_indices_2))+"\n"+str(top_pca_indices_2))
        top_pca_vals_2 = pc_ftr_corr[top_pca_indices_2]
        #top_indices.append()

    unique_indices = np.unique(top_indices)
    unique_vals = [left_singular_vectors[i,:] for i in unique_indices]
    unique_labels = [data_labels[i] for i in unique_indices]


    im, cbar = heatmap(np.array(unique_vals), unique_labels, range(num_modes), ax=ax, cbarlabel="contribution")
    texts = annotate_heatmap(im, np.array(unique_vals), valfmt="{x:.1f}")
    fig.tight_layout()
    plt.show()

    #ax.imshow(top_vals, cmap='bwr', vmin=-1, vmax=1, aspect='auto')


    fig.savefig(prefix+'_modes.png')

    fig.clear()
    plt.cla()
    plt.clf()
    print("done mode analysis")

def run_pca(data, data_label:str, data_labels, dims=10):

    if dims < 1:
        pca = pyemma.coordinates.pca(data, var_cutoff=dims)
    else:
        pca = pyemma.coordinates.pca(data, dim=dims)
    pca_output = pca.get_output()
    if dims < 1:
        n_dims = pca.ndim
    else:
        n_dims = dims
    prefix = 'pca_d'+str(n_dims)+'_'+data_label
    pca.save(prefix+'.model', overwrite=True)

    pc_ftr_corr = pca.feature_PC_correlation
    mode_ftr_analysis(pc_ftr_corr, prefix, data_labels, topN = 10, data_label=data_label)

    pca_concatenated = np.concatenate(pca_output)

    fig, ax, misc = pyemma.plots.plot_density(pca_concatenated[:,0], pca_concatenated[:,1], cbar=True, alpha=0.1)
    fig.savefig(prefix+'_density.png')
    print("done pca")

def run_tica(data, data_label:str, data_labels, dims=10, lag=10):
    if dims < 1:
        tica = pyemma.coordinates.tica(data, lag=lag, var_cutoff=dims)
    else:
        tica = pyemma.coordinates.tica(data, lag=lag, dim=dims)
    tica_output = tica.get_output()
    if dims < 1:
        n_dims = tica.ndim
    else:
        n_dims = dims
    prefix = 'tica_d'+str(n_dims)+'_'+'l_'+str(lag*ftr_timestep)+'ns_'+data_label
    tica.save(prefix+'.model', overwrite=True)

    tic_ftr_corr = tica.feature_TIC_correlation
    mode_ftr_analysis(tic_ftr_corr, prefix, data_labels, topN = 10, data_label=data_label)

    tica_concatenated = np.concatenate(tica_output)

    fig, ax, misc = pyemma.plots.plot_density(tica_concatenated[:,0], tica_concatenated[:,1], cbar=True, alpha=0.1)
    fig.savefig(prefix+'_density.png')
    print("done tica")

def run_vamp(data, data_label:str, data_labels, dims=10, lag=10):

    vamp = pyemma.coordinates.vamp(data, lag=lag, dim=dims)
    vamp_output = vamp.get_output()
    if dims < 1:
        n_dims = vamp.ndim
    else:
        n_dims = dims
    prefix = 'vamp_d'+str(n_dims)+'_'+'l_'+str(lag*ftr_timestep)+'ns_'+data_label
    vamp.save(prefix+'.model', overwrite=True)

    #mode_ftr_analysis(vamp.singular_vectors_left, prefix, data_labels, topN = 10, data_label=data_label)

    vamp_concatenated = np.concatenate(vamp_output)

    fig, ax, misc = pyemma.plots.plot_density(vamp_concatenated[:,0], vamp_concatenated[:,1], cbar=True, alpha=0.1)
    fig.savefig(prefix+'_density.png')
    print("done vamp")

# DATA VARIABLES

uni_dir = '/scratch365/mfarrugi/HMGR/universal-files'
ftrzn_dir = '/scratch365/mfarrugi/HMGR/500ns/analysis/pyemma-msm/1-featurization'
prmtop = md.load_prmtop(uni_dir+'/ts2-strip.prmtop')
files = glob.glob(ftrzn_dir+'/200ps/*.npy')

#lag_stride = 10 # load md to featurizers such that feature resolution is 1 ns or 1000 ps

# TORSIONS

#torsions = np.load(ftrzn_dir+'/200ps/ts2_torsn_ftrs.npy', allow_pickle=True)
#with open(ftrzn_dir+'/200ps/torsion_ftr_labels.txt', "rb") as f:
#    torsion_labels = pickle.load(f)
#torsions = list(torsions)
#torsions_concatenated = np.concatenate(torsions)

#print("torsions concat shape: "+str(np.shape(torsions_concatenated)))
#torsions_T = np.asarray(torsions_concatenated).T
#print("torsions_T \n"+str(torsions_T))
#print("shape "+str(np.shape(torsions_T)))
#print("lens "+str(len(torsions_T))+" "+str(len(torsions_T[0])))

#run_pca(torsions, 'torsions', data_labels = torsion_labels, dims=10)
#run_pca(torsions, 'torsions', data_labels= torsion_labels, dims=0.70)

#for dim_item in dim_list:
#    for lag_item in lag_list:
#        run_tica(torsions, 'torsions', data_labels = torsion_labels, dims=dim_item, lag=lag_item)
#        run_vamp(torsions, 'torsions', data_labels = torsion_labels, dims=dim_item, lag=lag_item)
#run_pca(torsions, 0.6, 'torsions', data_labels = torsion_labels)

# DISTANCES

distances = np.load(ftrzn_dir+'/200ps/ts2_jl2_ftrs.npy', allow_pickle=True)
with open(ftrzn_dir+'/200ps/jl2_ftr_labels.txt', "rb") as g:
    distance_labels = pickle.load(g)
distances = list(distances)
distances_concatenated = np.concatenate(distances)

distances_T = np.asarray(distances).T
#print("distances_T \n"+str(distances_T))
#print("shape "+str(np.shape(distances_T)))
#print("lens "+str(len(distances_T))+" "+str(len(distances_T[0])))#

run_pca(distances, 'jl2', data_labels = distance_labels, dims=10)
run_pca(distances, 'jl2', data_labels = distance_labels, dims=0.70)
for dim_item in dim_list:
    for lag_item in lag_list:
        run_tica(distances, 'jl2', data_labels = distance_labels, dims=dim_item, lag=lag_item)
        run_vamp(distances, 'jl2', data_labels = distance_labels, dims=dim_item, lag=lag_item)


