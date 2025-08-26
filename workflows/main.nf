#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { fetch_from_zenodo } from '../modules/fetch_from_zenodo.nf'
include { data_setup_synthetic } from '../modules/data_setup_synthetic.nf'
include { data_setup_zenodo } from '../modules/data_setup_zenodo.nf'
include { preprocessing    } from '../modules/preprocessing.nf'
include { evaluate_exactgp } from '../modules/evaluate_exactgp.nf'
include { evaluate_dkl     } from '../modules/evaluate_dkl.nf'
include { evaluate_rgasp   } from '../modules/evaluate_rgasp.nf'
include { evaluate_pca_rgasp } from '../modules/evaluate_pca_rgasp.nf'
include { benchmark_metrics } from '../modules/benchmark_metrics.nf'

workflow {
    println "▶ Starting pipeline with caseStudy=${params.caseStudy}"
    
    // 0. Determine dataset type. Based on the dataset type, different workflows are executed.
    def dataset_config = params.datasets[params.caseStudy]
    def dataset_type = dataset_config.type
    def data_ch

    if (dataset_type == "generate") {

        // 1. Data setup: Generate synthetic data directly
        data_ch = data_setup_synthetic(params.caseStudy)

        // 2. Preprocessing: Standardize, split, save to HDF5
        def tensors_ch = preprocessing(data_ch)

        // 3. Train and inference: Train emulators in parallel based on selected models and get predictions
        def exactgp_ch    = evaluate_exactgp(tensors_ch)
        def dkl_ch        = evaluate_dkl(tensors_ch)
        def rgasp_ch      = evaluate_rgasp(tensors_ch)
        def pca_rgasp_ch  = evaluate_pca_rgasp(tensors_ch)

        // 4. Benchmark: Compare metrics and save results
        benchmark_metrics(exactgp_ch, dkl_ch, rgasp_ch, pca_rgasp_ch)

    } else if (dataset_type == "zenodo") {

        // 1. Data setup: Fetch from Zenodo then process
        def raw_ch = fetch_from_zenodo(params.caseStudy)
        data_ch = data_setup_zenodo(raw_ch)

        // 2. Preprocessing: Standardize, split, save to HDF5
        def tensors_ch = preprocessing(data_ch)

        // 3. Train and inference: Train emulators in parallel based on selected models and get predictions
        // Real-world dataset is oftern larger, so exact inference may not be tractable
        // We only evaluate PCA-RGaSP and DKL
        def dkl_ch        = evaluate_dkl(tensors_ch)
        def pca_rgasp_ch  = evaluate_pca_rgasp(tensors_ch)

        // 4. Benchmark: Compare metrics and save results
        // Use distinct NO_FILE placeholders to avoid filename collision
        def nofile_exactgp = file("${workflow.launchDir}/assets/NO_FILE_EXACTGP", checkIfExists:true)
        def nofile_rgasp   = file("${workflow.launchDir}/assets/NO_FILE_RGASP", checkIfExists:true)
        def exactgp_stub = Channel.of(nofile_exactgp)
        def rgasp_stub   = Channel.of(nofile_rgasp)
        benchmark_metrics(exactgp_stub, dkl_ch, rgasp_stub, pca_rgasp_ch)

    } else {
        error "Unsupported dataset type: ${dataset_type}"
    }
}
