process benchmark_metrics {
  tag "benchmark_metrics"
  publishDir "results", mode: 'copy'
  
  input:
    path exactgp_results
    path dkl_results
    path rgasp_results

  output:
    path "benchmark_results", emit: result_dir

  script:
  """
  python ${workflow.launchDir}/scripts/benchmark_metrics.py \
    --exactgp-metrics-dir ${exactgp_results} \
    --dkl-metrics-dir ${dkl_results} \
    --rgasp-metrics-dir ${rgasp_results} \
    --output-dir benchmark_results
  """
}
