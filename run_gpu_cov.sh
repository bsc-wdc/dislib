#!/bin/bash -e

module purge
module load bullxmpi/bullxmpi-1.2.9.1 COMPSs/TrunkCT
module load gcc/9.2.0 cuda/10.1 mkl/2018.1 ANACONDA/2021.05
module unload python

eval "$(conda shell.bash hook)"
conda activate cupy-cuda101

export PYTHONPATH=/apps/COMPSs/3.1/Bindings/python/3/

DISLIB_GPU_AVAILABLE=True python3 -m coverage run --data-file=gpu_cov --source dislib tests