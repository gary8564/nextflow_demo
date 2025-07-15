#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { data_setup       } from '../modules/data_setup.nf'
include { preprocessing    } from '../modules/preprocessing.nf'
include { train_exactgp    } from '../modules/train_exactgp.nf'
include { train_dkl        } from '../modules/train_dkl.nf'
include { evaluate_metrics } from '../modules/evaluate_metrics.nf'

workflow {
    println "▶ Starting pipeline with caseStudy=${params.caseStudy}"
    // 1. generate or fetch raw X/Y
    def raw     = data_setup( params.caseStudy )
    // 2. standardize, split, tensorize
    def tensors = preprocessing( raw )
    // 3. train two emulators in parallel
    def exactgp = train_exactgp( tensors )
    def dkl     = train_dkl( tensors )
    // 4. compare metrics
    evaluate_metrics( exactgp, dkl )
}
