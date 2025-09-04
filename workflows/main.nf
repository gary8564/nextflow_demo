#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { fetch_from_zenodo } from '../modules/fetch_from_zenodo.nf'
include { data_setup_synthetic } from '../modules/data_setup_synthetic.nf'
include { data_setup_tsunami } from '../modules/data_setup_tsunami.nf'
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
    def dataset_source = dataset_config.source
    def data_ch

    if (dataset_source == "") {
        // 1. Data setup: Generate synthetic data directly
        data_ch = data_setup_synthetic(params.caseStudy)

        // 2. Preprocessing: Standardize, split, save to HDF5
        def processed_ch = preprocessing(data_ch)

        // 3. Train and inference: Train emulators in parallel based on selected models and get predictions
        def exactgp_ch   = evaluate_exactgp(processed_ch)
        def dkl_ch       = evaluate_dkl(processed_ch)
        def rgasp_ch     = evaluate_rgasp(processed_ch)
        def pca_rgasp_ch = evaluate_pca_rgasp(processed_ch)

        // 4. Gather result directories from all evaluated models and benchmark
        def metrics_list_ch = exactgp_ch.exactgp
                                .mix(dkl_ch.dkl)
                                .mix(rgasp_ch.rgasp)
                                .mix(pca_rgasp_ch.pca_rgasp)
                                .map { path -> path.toString() }
                                .collect()

        benchmark_metrics(metrics_list_ch)

    } else if (dataset_source == "zenodo") {
        // 1. Data setup: Fetch from Zenodo then process
        def raw_ch = fetch_from_zenodo(params.caseStudy)
        data_ch = data_setup_tsunami(raw_ch)

        // 2. Preprocessing: Standardize, split, save to HDF5
        def processed_ch = preprocessing(data_ch)

        // 3. Train and inference: For real-world tsunami data,  RGaSP on ultra-high-dimensional inputs would face numerical instability issues
        // Evaluate ExactGP, DKL and PCA-RGaSP
        def exactgp_ch   = evaluate_exactgp(processed_ch)
        def dkl_ch       = evaluate_dkl(processed_ch)
        def pca_rgasp_ch = evaluate_pca_rgasp(processed_ch)

        // 4. Gather result directories and benchmark
        def metrics_list_ch = exactgp_ch.exactgp
                                    .mix(dkl_ch.dkl)
                                    .mix(pca_rgasp_ch.pca_rgasp)
                                    .map { path -> path.toString() }
                                    .collect()
        benchmark_metrics(metrics_list_ch)
        
    } else {
        error "NotImplementedError: ${dataset_source} is not supported."
    }
}
