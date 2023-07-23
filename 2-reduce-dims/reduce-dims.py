#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib as mpl
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

config.show_progress_bars = False

def mode_ftr_analysis(ftr_corr, prefix:str, data_labels, cutoff0 = 0.9, cutoff=0.75, topN=None, data_label='data'):
    
    num_modes = len(ftr_corr[0])
    
    fig,ax = plt.subplots()
    
    top_indices = []
    top_vals = []
    if topN is not None:
        for m in range(num_modes):
            top_m = map((heapq.nlargest(topN, enumerate(ftr_corr[:,m]), key=lambda x: x[1])))
            top_indices.append()
    else:
        top_indices.append()

    top_pca_indices_1 = np.vstack((np.argwhere(ftr_corr[:,0] > 0.95), np.argwhere(pc_ftr_corr[:,0] < -0.95)))
    print("top pca indices len "+str(len(top_pca_indices_1))+"\n"+str(top_pca_indices_1))
    top_pca_vals_1 = pc_ftr_corr[top_pca_indices_1]

    top_pca_indices_2 = np.vstack((np.argwhere(pc_ftr_corr[:,1] > 0.8), np.argwhere(pc_ftr_corr[:,1] < -0.8)))
    print("top pca indices len "+str(len(top_pca_indices_2))+"\n"+str(top_pca_indices_2))
    top_pca_vals_2 = pc_ftr_corr[top_pca_indices_2]

    top_pca_vals = np.vstack((top_pca_vals_1, top_pca_vals_2))

    i = ax.imshow(top_pca_vals, cmap='bwr', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel('PC')

    #ax.set_yticks(top_pca_indices)
    ax.set_ylabel('torsion')


    fig.colorbar(i);
    fig.savefig('pca_corr_matrix_ftrs.png')

    top_torsions_1 = [torsion_labels[i] for i in top_pca_indices_1[:,0]]
    top_torsions_2 = [torsion_labels[i] for i in top_pca_indices_2[:,0]]

    fig.clear()
    plt.cla()
    plt.clf()



def run_pca(data, dims, data_label:str, data_labels):

    pca = pyemma.coordinates.pca(torsions, dim=d)
    pca_output = pca.get_output()
    if dims < 1:
        n_dims = pca.ndims
    else:
        n_dims = dims
    prefix = 'pca_d'+str(n_dims)+'_'+data_label+'_'

    pc_ftr_corr = pca.feature_PC_correlation
    mode_ftr_analysis(pc_ftr_corr, prefix, data_labels, topN = 10, data_label=data_label)


def run_tica()


def run_vamp()


# DATA VARIABLES

uni_dir = '/scratch365/mfarrugi/HMGR/universal-files'
ftrzn_dir = '/scratch365/mfarrugi/HMGR/500ns/analysis/pyemma-msm/1-featurization'
prmtop = md.load_prmtop(uni_dir+'/ts2-strip.prmtop')
files = glob.glob(ftrzn_dir+'/200ps/*.npy')

ftr_timestep = 0.200 # iniital md resolution of 50 ps, or 0.05 ns, ftrzn stride of 4, so 200ps or 0.2ns resolution
lag_stride = 10 # load md to featurizers such that feature resolution is 1 ns or 1000 ps

# TORSIONS

torsions = np.load(ftrzn_dir+'/200ps/ts2_torsn_ftrs.npy', allow_pickle=True)
with open(ftrzn_dir+'/200ps/torsion_ftr_labels.txt', "rb") as f:
    torsion_labels = pickle.load(f)
torsions = list(torsions)
torsions_concatenated = np.concatenate(torsions)

#print("torsions concat shape: "+str(np.shape(torsions_concatenated)))
torsions_T = np.asarray(torsions_concatenated).T
#print("torsions_T \n"+str(torsions_T))
#print("shape "+str(np.shape(torsions_T)))
#print("lens "+str(len(torsions_T))+" "+str(len(torsions_T[0])))



# DISTANCES

distances = np.load(ftrzn_dir+'/200ps/ts2_jl2_ftrs.npy', allow_pickle=True)
with open(ftrzn_dir+'/200ps/jl2_ftr_labels.txt', "rb") as g:
    distance_labels = pickle.load(g)
distances = list(distances)
distances_concatenated = np.concatenate(distances)

distances_T = np.asarray(distances).T
print("distances_T \n"+str(distances_T))
print("shape "+str(np.shape(distances_T)))
print("lens "+str(len(distances_T))+" "+str(len(distances_T[0])))



