process benchmark_metrics {
  tag "benchmark_metrics"
  publishDir "${params.outDir}/${params.caseStudy}", mode: 'copy'
  
  input:
    path exactgp_results
    path dkl_results
    path rgasp_results
    path pca_rgasp_results

  output:
    path "benchmark_results", emit: result_dir

  script:
  def isPlaceholder = { f -> f.name.startsWith('NO_FILE') }
  def exactgp_arg   = !isPlaceholder(exactgp_results)   ? "--exactgp-metrics-dir ${exactgp_results}" : ''
  def dkl_arg       = !isPlaceholder(dkl_results)       ? "--dkl-metrics-dir ${dkl_results}" : ''
  def rgasp_arg     = !isPlaceholder(rgasp_results)     ? "--rgasp-metrics-dir ${rgasp_results}" : ''
  def pca_rgasp_arg = !isPlaceholder(pca_rgasp_results) ? "--pca-rgasp-metrics-dir ${pca_rgasp_results}" : ''
  """
  python ${workflow.launchDir}/scripts/benchmark_metrics.py \
    ${exactgp_arg} \
    ${dkl_arg} \
    ${rgasp_arg} \
    ${pca_rgasp_arg} \
    --output-dir benchmark_results
  """
}
