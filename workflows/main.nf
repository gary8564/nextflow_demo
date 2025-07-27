#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { fetch_data       } from '../modules/fetch_data.nf'
include { data_setup       } from '../modules/data_setup.nf'
include { preprocessing    } from '../modules/preprocessing.nf'
include { train_exactgp    } from '../modules/train_exactgp.nf'
include { train_dkl        } from '../modules/train_dkl.nf'
include { evaluate_metrics } from '../modules/evaluate_metrics.nf'

workflow {
    println "▶ Starting pipeline with caseStudy=${params.caseStudy}"
    // 1. fetch raw inputs (download or generate metadata)
    def raw_ch = fetch_data( params.caseStudy )
    // 2. convert raw -> X/Y
    def data_ch = data_setup( raw_ch )
    // 3. standardize, split, tensorize
    def tensors_ch = preprocessing( data_ch )
    // 4. train two emulators in parallel
    def exactgp_ch = train_exactgp( tensors_ch )
    def dkl_ch     = train_dkl( tensors_ch )
    // 5. compare metrics and save results
    evaluate_metrics( exactgp_ch, dkl_ch )
}
