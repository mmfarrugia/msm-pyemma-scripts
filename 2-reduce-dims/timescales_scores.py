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

timesteps = ftr_timestep*lag_list