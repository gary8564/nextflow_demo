process benchmark_metrics {
  tag "benchmark_metrics"
  publishDir "${params.outDir}/${params.caseStudy}", mode: 'copy'
  
  input:
    val metrics_paths

  output:
    path "benchmark_results", emit: result_dir

  script:
  // Convert Groovy list [path1, path2, path3, ...] to JSON string ["path1", "path2", "path3", ...]
  def metrics_list = groovy.json.JsonOutput.toJson(metrics_paths)
  """
  python ${workflow.launchDir}/scripts/benchmark_metrics.py \
    --metrics-paths '${metrics_list}' \
    --output-dir benchmark_results
  """
}
