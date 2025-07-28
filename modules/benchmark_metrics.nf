process benchmark_metrics {
  tag "benchmark_metrics"
  input:
    path exactgp
    path dkl

  output:
    path "${params.outDir}", emit: result_dir

  script:
  """
  python ${workflow.launchDir}/scripts/benchmark_metrics.py \
    --exact-metrics ${exactgp} \
    --dkl-metrics   ${dkl} \
    --output-dir    ${params.outDir}
  """
}
