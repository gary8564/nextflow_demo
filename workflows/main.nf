#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { fetch_data       } from '../modules/fetch_data.nf'
include { data_setup       } from '../modules/data_setup.nf'
include { preprocessing    } from '../modules/preprocessing.nf'
include { evaluate_exactgp } from '../modules/evaluate_exactgp.nf'
include { evaluate_dkl     } from '../modules/evaluate_dkl.nf'
include { benchmark_metrics } from '../modules/benchmark_metrics.nf'

workflow {
    println "▶ Starting pipeline with caseStudy=${params.caseStudy}"
    // 1. fetch raw inputs (download or generate metadata)
    def raw_ch = fetch_data( params.caseStudy )
    // 2. convert raw -> X/Y
    def data_ch = data_setup( raw_ch )
    // 3. standardize, split, tensorize
    def tensors_ch = preprocessing( data_ch )
    // 4. train two emulators in parallel
    def exactgp_ch = evaluate_exactgp( tensors_ch )
    def dkl_ch     = evaluate_dkl( tensors_ch )
    // 5. compare metrics and save results
    benchmark_metrics( exactgp_ch, dkl_ch )
}
