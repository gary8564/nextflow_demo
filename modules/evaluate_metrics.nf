process evaluate_metrics {
  tag "evaluate_metrics"
  input:
    path exactgp
    path dkl

  output:
    path "${params.outDir}", emit: result_dir

  script:
  """
  python3 ${workflow.launchDir}/scripts/evaluate_metrics.py \
    --exact-metrics ${exactgp} \
    --dkl-metrics   ${dkl} \
    --output-dir    ${params.outDir}
  """
}
