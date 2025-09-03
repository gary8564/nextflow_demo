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
        def processed_ch = preprocessing(data_ch)

        // 3. Train and inference: Train emulators in parallel based on selected models and get predictions
        def exactgp_ch   = evaluate_exactgp(processed_ch)
        def dkl_ch       = evaluate_dkl(processed_ch)
        def rgasp_ch     = evaluate_rgasp(processed_ch)
        def pca_rgasp_ch = evaluate_pca_rgasp(processed_ch)

        // 4. Gather metrics from all evaluated models and benchmark
        def exactgp_metrics = exactgp_ch.exactgp.map { dir -> new groovy.json.JsonSlurper().parse(new File("${dir}/metrics.json")) }
        def dkl_metrics     = dkl_ch.dkl.map       { dir -> new groovy.json.JsonSlurper().parse(new File("${dir}/metrics.json")) }
        def rgasp_metrics   = rgasp_ch.rgasp.map   { dir -> new groovy.json.JsonSlurper().parse(new File("${dir}/metrics.json")) }
        def pca_metrics     = pca_rgasp_ch.pca_rgasp.map { dir -> new groovy.json.JsonSlurper().parse(new File("${dir}/metrics.json")) }

        def metrics_list_ch = exactgp_metrics 
                                .mix(dkl_metrics)
                                .mix(rgasp_metrics)
                                .mix(pca_metrics)
                                .collect()

        benchmark_metrics(metrics_list_ch)

    } else if (dataset_type == "zenodo") {
        // 1. Data setup: Fetch from Zenodo then process
        def raw_ch = fetch_from_zenodo(params.caseStudy)
        data_ch = data_setup_zenodo(raw_ch)

        // 2. Preprocessing: Standardize, split, save to HDF5
        def processed_ch = preprocessing(data_ch)

        // 3. Train and inference: Train emulators in parallel based on selected models and get predictions
        // Real-world dataset is oftern larger, so exact inference may not be tractable
        // We only evaluate PCA-RGaSP and DKL
        def dkl_ch       = evaluate_dkl(processed_ch)
        def pca_rgasp_ch = evaluate_pca_rgasp(processed_ch)

        // 4. Gather metrics and benchmark (DKL + PCA-RGaSP only)
        def dkl_metrics = dkl_ch.dkl.map { dir -> new groovy.json.JsonSlurper().parse(new File("${dir}/metrics.json")) }
        def pca_metrics = pca_rgasp_ch.pca_rgasp.map { dir -> new groovy.json.JsonSlurper().parse(new File("${dir}/metrics.json")) }
        def metrics_list_ch = dkl_metrics.mix(pca_metrics).collect()
        benchmark_metrics(metrics_list_ch)
        
    } else {
        error "Unsupported dataset type: ${dataset_type}"
    }
}
