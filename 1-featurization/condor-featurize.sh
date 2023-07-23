#! /bin/bash
date
hostname
if [ -r /opt/crc/Modules/current/init/bash ]; then
    source /opt/crc/Modules/current/init/bash
fi

echo "- Entering scratch365 submit directory"
cd $_CONDOR_JOB_IWD
echo "- Executing pmemd.cuda"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/crc/c/conda/2022/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/crc/c/conda/2022/etc/profile.d/conda.sh" ]; then
        . "/opt/crc/c/conda/2022/etc/profile.d/conda.sh"
    else
        export PATH="/opt/crc/c/conda/2022/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate pyemma-plus

conda list

which python

#cd /scratch365/mfarrugi/HMGR/500ns/analysis

python pyemma-featurize.py > pyemma-featurize.out


