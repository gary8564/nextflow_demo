#!/bin/bash -ue
python3 /Users/garychang/Documents/RWTH/HiWiJob/MBD/nextflow_demo/scripts/evaluate_metrics.py     --exact-metrics results_exactgp/metrics.json     --dkl-metrics   results_dkl/metrics.json     --output-dir    results
