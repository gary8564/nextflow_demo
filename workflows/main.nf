#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { fetch_from_zenodo } from '../modules/fetch_from_zenodo.nf'
include { data_setup_synthetic } from '../modules/data_setup_synthetic.nf'
include { data_setup_zenodo } from '../modules/data_setup_zenodo.nf'
include { preprocessing    } from '../modules/preprocessing.nf'
include { evaluate_exactgp } from '../modules/evaluate_exactgp.nf'
include { evaluate_dkl     } from '../modules/evaluate_dkl.nf'
include { evaluate_rgasp   } from '../modules/evaluate_rgasp.nf'
include { benchmark_metrics } from '../modules/benchmark_metrics.nf'

workflow {
    println "▶ Starting pipeline with caseStudy=${params.caseStudy}"
    
    // Determine dataset type and handle accordingly
    def dataset_config = params.datasets[params.caseStudy]
    def dataset_type = dataset_config.type
    
    // 1. Data setup based on dataset type
    def data_ch
    if (dataset_type == "generate") {
        // Generate synthetic data directly
        data_ch = data_setup_synthetic(params.caseStudy)
    } else if (dataset_type == "zenodo") {
        // Fetch from Zenodo then process
        def raw_ch = fetch_from_zenodo(params.caseStudy)
        data_ch = data_setup_zenodo(raw_ch)
    } else {
        error "Unsupported dataset type: ${dataset_type}"
    }
    
    // 2. Standardize, split, save to HDF5
    def tensors_ch = preprocessing(data_ch)
    
    // 3. Train emulators in parallel based on selected models
    def exactgp_ch = evaluate_exactgp(tensors_ch)
    def dkl_ch     = evaluate_dkl(tensors_ch)
    def rgasp_ch   = evaluate_rgasp(tensors_ch)
    
    // 4. Compare metrics and save results
    benchmark_metrics(exactgp_ch, dkl_ch, rgasp_ch)
}
