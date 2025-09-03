process benchmark_metrics {
  tag "benchmark_metrics"
  publishDir "${params.outDir}/${params.caseStudy}", mode: 'copy'
  
  input:
    val metrics_list

  output:
    path "benchmark_results", emit: result_dir

  script:
  def metrics_json = groovy.json.JsonOutput.toJson(metrics_list)
  """
  cat > metrics_list.json <<'JSON'
  ${metrics_json}
  JSON

  python ${workflow.launchDir}/scripts/benchmark_metrics.py \
    --metrics-file metrics_list.json \
    --output-dir benchmark_results
  """
}
