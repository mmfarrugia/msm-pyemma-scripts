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
ftr_timestep = 0.200 # ns
lag_list = np.array([5, 10, 50, 500, 1000, 2000, 2400])
dim_list = [2, 4, 10]

redn_dir = '/scratch365/mfarrugi/HMGR/500ns/analysis/msm_pyemma_scripts/2-reduce-dims/'
ftrzn_dir = '/scratch365/mfarrugi/HMGR/500ns/analysis/msm_pyemma_scripts/1-featurization/'

ftrzn_files = glob.glob(ftrzn_dir+'/200ps/torsions/*.npy')

# set source
source = pyemma.coordinates.source(ftrzn_files, allow_pickle=True)

source_name = 'torsions'

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

def generate_dim_labels(label_list:list, n_dims:int):
    labels = [label+str(i) for label in label_list for i in range(n_dims)]
    print(labels)
    return labels

def generate_prefix(data_label:str, dim:int, timestep:int, model_type:str):
    return model_type+'_d'+str(dim)+'_l_'+str(timestep)+'ns_'+data_label

def compare_reductions(prefix:str, pca_model, tica_model, vamp_model, n_dims:int):
    if n_dims > 5: n_dims = 5 
    fig, axes = plt.subplots(figsize=(10, 10))
    pca_output = pca_model.get_output()
    pca_concatenated = np.concatenate(pca_output)
    tica_output = tica_model.get_output()
    tica_concatenated = np.concatenate(tica_output)
    vamp_output = vamp_model.get_output()
    vamp_concatenated = np.concatenate(vamp_output)
    concat = np.concatenate([pca_concatenated[:,:n_dims], tica_concatenated[:,:n_dims], vamp_concatenated[:,:n_dims]], axis=1)
    print("shape concat: "+str(np.shape(concat)))
    pyemma.plots.plot_feature_histograms(concat, feature_labels= generate_dim_labels(['PCA', 'TICA', 'VAMP'], n_dims=n_dims), ax=axes)
    fig.tight_layout()
    fig.savefig(prefix+'_ftr_hist.png')
    plt.close()

def dimensional_analysis(prefix:str, is_tica:bool, models, num_modes:int):
    if num_modes > 15: num_modes = 15
    fig, ax = plt.subplots()
    if is_tica:
        for i, timestep in enumerate(timesteps):
            color = 'C{}'.format(i)
            #ax.fill_between(range(num_modes), models[i].timescales, )
            timescales = models[i].timescales[:num_modes]
            scaled_timescales = timescales * ftr_timestep
            print(str(scaled_timescales))
            ax.plot(range(num_modes), scaled_timescales, '--o', color=color, label='lag={:.1f}ns'.format(timesteps[i]))
        ax.set_xlabel('# of dimensions/modes')

    else:
#        print('dim_models list\n'+str([dim_models for dim_models in models]))
#        print('nested\n'+str([[dim_models[i] for i in range(len(timesteps))] for dim_models in models]))
        scores = np.array([[dim_models[i].score() for i in range(len(timesteps))] for dim_models in models])
        print(str(np.shape(scores)))
        for i, timestep in enumerate(timesteps):
            color = 'C{}'.format(i)
            ax.plot(dim_list, scores[:,i], '--o', color=color, label='lag={:.1f}ns'.format(timesteps[i]))
            ax.set_xlabel("lag (ns)")

    ax.legend()
    ax.set_ylabel('Implied Timescales (ns)' if is_tica else 'VAMP2 Score')
    fig.tight_layout()
    fig.savefig(prefix+'_dim_analysis.png')
    plt.close()
    return -1
        

def lag_analysis(model):
    for i, timestep in enumerate(timesteps):
        print()
    return -1 # Return index of chosen lag time within timesteps/lag_list


def mode_densities(prefix:str, model, num_modes:int):
    # density maps of different combinations of modes
    if num_modes > 5: num_modes = 5
    model_output = model.get_output()
    concatenated = np.concatenate(model_output)
    fig, axes = plt.subplots(num_modes, num_modes-1, figsize=(15,12))
    for i in range(num_modes):
        for j in range(num_modes-i):
            pyemma.plots.plot_density(concatenated[:,i], concatenated[:,j], ax=axes[i,j-1], cbar=True, alpha=0.1)
    fig.tight_layout()
    fig.savefig(prefix+'_IC_densities.png')
    plt.close()

def traj_IC_hist(prefix:str, model, num_modes:int):
    if num_modes > 5: num_modes = 5
    model_output = model.get_output()
    concat = np.concatenate(model_output)
    fig, ax = pyemma.plots.plot_feature_histograms(concat[:,:num_modes])
    for i, mode in enumerate(['IC']*num_modes):
        ax.plot(concat[:, num_modes - 1 - i], np.linspace(-0.2+i,0.8+i,50000), color='C2', alpha=0.6)
        ax.annotate('${}$(time)'.format(mode), xy=(3, 0.6 + i), xytext=(3, i),
                    arrowprops=dict(fc='C2', ec='None', alpha=0.6, width=2))
        
    fig.tight_layout()
    fig.savefig(prefix+'_traj_IC_hist.png')
    plt.close()

def subspace_timeseries(prefix:str, pca_concatenated, tica_concatenated, vamp_concatenated):
    #NOTE: concatenated input should also just be within the subspace to visualize, so already a singular dimension's values (vector basically)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(pca_concatenated, label='PCA')
    ax.plot(tica_concatenated, label='TICA')
    # check directionality of VAMP for easier visualization
    ax.plot(vamp_concatenated, label='VAMP')
    ax.set_xlabel('time / steps')
    ax.set_ylabel('mode values')
    ax.legend()
    fig.tight_layout()
    fig.savefig(prefix+'_subspace_timeseries.png')
    plt.close()

def topN_features_hist(prefix:str):
    print()

def load_model(data_label:str, dim:int, timestep:int, model_type:str):
    prefix = generate_prefix(data_label=data_label, dim=dim, timestep=time, model_type=model_type)
    model = pyemma.load(redn_dir+prefix+'.model')
    model.data_producer = source
    return model, prefix

def load_models(data_label:str, dims=None, times=None, model_types=None):
    if dims is None:
        dims = dim_list
    if times is None:
        times=timesteps
    if model_types is None:
        model_types = ['pca', 'tica', 'vamp']
    models = dict()
    for dim in dims:
        for time in times:
            for model_type in model_types:
                model, prefix = load_model(data_label=data_label, dim=dim, timestep=time, model_type=model_type)
                models[prefix] = model
    return models

def load_all_models():
    files = glob.glob(redn_dir+'/*.model')
    models = dict()
    for file in files:
        model =pyemma.load(file)
        model.data_producer = source
        models[file[:-6]] = model
    return models

# Load pca model
pcaprefix = 'pca_d10_'+source_name
pca = pyemma.load(redn_dir+pcaprefix+'.model')
pca.data_producer = source

# Mode Cross-sections Analysis
mode_densities(pcaprefix, pca, 4)

# Load tICA and VAMP models at all timesteps
tica = list()
vamp = list()
for timestep in timesteps:
    param_prefix = 'd10_l_'+str(timestep)+'ns_'+source_name
    tica_prefix = 'tica_'+param_prefix
    tica_model_ = pyemma.load(redn_dir+tica_prefix+'.model')
    tica_model_.data_producer = source
    tica.append(tica_model_) 
    vamp_prefix = 'vamp_'+param_prefix
    vamp_model_ = pyemma.load(redn_dir+vamp_prefix+'.model')
    vamp_model_.data_producer = source
    vamp.append(vamp_model_) 

    # Might as well run the analyses which require iterating
    compare_reductions(param_prefix, pca, tica_model_, vamp_model_, n_dims=4)
    mode_densities(tica_prefix, tica_model_, 4)
    mode_densities(vamp_prefix, vamp_model_, 4)


# Dimensional Analysis

#pca_dim = dimensional_analysis('pca_d10', False, pca, 10)
tica_dim = dimensional_analysis('tica_d10_'+source_name, True, tica, 10)
def run_vamp_dim_analysis():
    vamp_all = list()
    for dim in dim_list:
        vamp_dim = list()
        for timestep in timesteps:
            param_prefix = 'd'+str(dim)+'_l_'+str(timestep)+'ns_'+source_name
            vamp_prefix = 'vamp_'+param_prefix
            vamp_model_ = pyemma.load(redn_dir+vamp_prefix+'.model')
            vamp_model_.data_producer = source
            vamp_dim.append(vamp_model_)
        vamp_all.append(vamp_dim)

    vamp_dim = dimensional_analysis('vamp_'+source_name, False, vamp_all, 10)

# Lag Analysis

#tica_lag = lag_analysis()
#vamp_lag = lag_analysis()


# IC histogram of time spent with trajectory paths overlaid

pca_concat = np.concatenate(pca.get_output())

traj_IC_hist(pcaprefix, pca, 4) 
traj_IC_hist('tica_d10_'+source_name, tica[5], 4)
traj_IC_hist('vamp_d10_'+source_name, vamp[4], 4)

tica_5 = tica[5].get_output()
tica_5 = np.concatenate(tica_5)
tic_1 = tica_5[:,0]

vamp_5 = vamp[5].get_output()
vamp_5 = np.concatenate(vamp_5)

subspace_timeseries('400ns_dim_1_of_10_torsion_', pca_concat[:,0], tica_5[:,0], vamp_5[:,0])

# traj_IC_hist(pcaprefix, pca, pca_dim) 
# traj_IC_hist(tica_prefix, tica[tica_lag], tica_dim)
# traj_IC_hist(vamp_prefix, vamp[vamp_lag], vamp_dim)
