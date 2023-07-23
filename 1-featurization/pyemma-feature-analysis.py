#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mdshare
import pyemma
from pyemma.util.contexts import settings
import mdtraj as md
import glob
import deeptime
import multiprocessing
import time
from threading import Thread

class SplitThread(Thread):
    def __init__(self, args):
        Thread.__init__(self)
        #set default value
        self.args = args
        self.score = -1.0
    #function executed in a new thread
    def run(self):
        data, nval, lag, dim = self.args
        ival = np.random.choice(len(data), size=nval, replace=False)
        vamp = deeptime.decomposition.VAMP(lagtime=lag, dim=dim)
        print("vamp instance")
        covars = deeptime.decomposition.VAMP.covariance_estimator(lagtime=lag).fit(
            [d for i, d in enumerate(data) if i not in ival]).fetch_model()
        print("covars fetched")
        vamp.fit(covars)
        print("vamp fit to covars")
        model = vamp.fetch_model()
        print("model fetched")
        self.score = model.score(r=2)
        print("model score "+str(self.score))
# MULTITHREADED ATTEMPT
class ScoresThread(Thread):
    def __init__(self, args):
        Thread.__init__(self)
        #set default value
        self.args = args
        self.scores = np.zeros(10)
    #function executed in a new thread
    def run(self):
        self.scores = score_cv(self.args[0], self.args[1], self.args[2], number_of_splits=10)
# initial resolution 50ps, resolution of 00 pentapeptide tutorial was 0.1 ns so decided on stride of 2 ^ to bring to 0.1 ns resolution

def score_cv(data, dim, lag, number_of_splits=10, validation_fraction=0.5):
    """Compute a cross-validated VAMP2 score.
    We randomly split the list of independent trajectories into
    a training and a validation set, compute the VAMP2 score,
    and repeat this process several times.
    Parameters
    ----------
    data : list of numpy.ndarrays
        The input data.
    dim : int
        Number of processes to score; equivalent to the dimension
        after projecting the data with VAMP2.
    lag : int
        Lag time for the VAMP2 scoring.
    number_of_splits : int, optional, default=10
        How often do we repeat the splitting and score calculation.
    validation_fraction : int, optional, default=0.5
        Fraction of trajectories which should go into the validation
        set during a split.
    """
    # we temporarily suppress very short-lived progress bars
    print("score cv start")
    #with pyemma.util.contexts.settings(show_progress_bars=False):
    nval = int(len(data) * validation_fraction)
    scores = np.zeros(number_of_splits)
    subthreads = [None]*number_of_splits
    for n in range(number_of_splits):
        subthreads[n] = SplitThread(args=(data, nval, lag, dim))
        print("n "+str(n))
        subthreads[n].start()
    while(True in [subthread.is_alive() for subthread in subthreads]):
        time.sleep(30)
    print("subthreads "+str(len(data))+" completed")
    scores = [subthread.score for subthread in subthreads]
    print("score cv end")
    return scores

#MAIN CODE

print("num cpus: ", multiprocessing.cpu_count())
uni_dir = '/scratch365/mfarrugi/HMGR/universal-files'
ftrzn_dir = '/scratch365/mfarrugi/HMGR/500ns/analysis/pyemma-msm/1-featurization'
prmtop = md.load_prmtop(uni_dir+'/ts2-strip.prmtop')
files = glob.glob(ftrzn_dir+'/200ns/*.npy')

torsions = np.load('ts2_torsn_ftrs.npy', allow_pickle=True)
backbone = np.load('ts2_backbone_xyz_ftrs.npy', allow_pickle=True)
distances = np.load('ts2_jl1_ftrs.npy', allow_pickle=True)
distances2 = np.load('ts2_jl2_ftrs.npy', allow_pickle=True)


# In[ ]:
dim = 10
variational_cutoff = 0.9
fig, axes = plt.subplots(1, 5, figsize=(20, 10), sharey=True)
lags = [5, 10, 20, 1000, 4000]
for ax, lag in zip(axes.flat, lags):
    print("lag "+str(lag))
    threads = [None]*4
    j = 0
    threads[j] = ScoresThread(args=(distances, dim, lag))
    print("thread start distances lag "+str(lag))
    threads[j].start()
    j+=1
    threads[j] = ScoresThread(args=(distances2, dim, lag))
    print("thread start distances lag "+str(lag))
    threads[j].start()
    j+=1
    #threads_status = [thread.is_alive() for thread in threads[0:j-1]]
    #threads[j-1].join()
    #print("midway joined")
    #while(True in threads_status):
    #    time.sleep(10)
    #    threads_status = [thread.is_alive() for thread in threads[0:j-1]]
    #print("midway threads completed")
    threads[j] = ScoresThread(args=(torsions_data, dim, lag))
    print("thread start torsions lag "+str(lag))
    threads[j].start()
    j+=1
    threads[j] = ScoresThread(args=(positions_data, dim, lag))
    print("thread start positions lag "+str(lag))
    threads[j].start() 
    j+=1
    threads_status = [thread.is_alive() for thread in threads]
    threads[-1].join()
    print("joined")
    while(True in threads_status):
        time.sleep(10)
        threads_status = [thread.is_alive() for thread in threads]
    print("threads completed")
    scores_ = np.array([thread.scores for thread in threads])
    print(scores_)
    scores = np.mean(scores_, axis=1)
    errors = np.std(scores_, axis=1, ddof=1)
    print(scores)
    print(errors)
    print(labels)
    ax.bar(labels, scores, yerr=errors)
    ax.set_title(r'lag time $\tau$={:.1f}ns'.format(lag * 0.1))
    if lag == 5:
        # save for later
        vamp_bars_plot = dict(
            labels=labels, scores=scores, errors=errors, dim=dim, lag=lag)
axes[0].set_ylabel('VAMP2 score')
fig.tight_layout()
print("about to save fig")
fig.savefig('feature_analysis.png')
# In[ ]:
# In[ ]:
# for lag times > ??ns, using more than ? dims -/-> increase score, just plateaus
# first four dims contain all relevant info of slow dynamics
# try a TICA projection with lag time ??ns (? steps)
