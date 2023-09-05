#!/bin/bash

#$ -M mfarrugi@nd.edu
#$ -m abe
#$ -pe smp 24
#$ -q debug      
#$ -N analyze-dims


conda activate pyemma-plus

which python

cd /scratch365/mfarrugi/HMGR/500ns/analysis/msm_pyemma_scripts/2-reduce-dims

python analyze-all-dims.py > analyze-all-dims.out
