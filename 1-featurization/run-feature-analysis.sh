#!/bin/bash

#$ -M mfarrugi@nd.edu
#$ -m abe
#$ -pe smp 64
#$ -q long            
#$ -N featurize


conda activate pyemma-plus

which python

cd /scratch365/mfarrugi/HMGR/500ns/analysis/pyemma-msm/1-featurization

python pyemma-featurize.py > pyemma-featurize-output.txt
