#!/bin/bash -ue
export KMP_DUPLICATE_LIB_OK=TRUE

python3 /Users/garychang/Documents/RWTH/HiWiJob/MBD/nextflow_demo/scripts/train_dkl.py     --input-dir data_tensors     --output-dir results_dkl
