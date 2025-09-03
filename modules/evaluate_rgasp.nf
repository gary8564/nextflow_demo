process evaluate_rgasp {
  conda "${workflow.launchDir}/envs/evaluate_rgasp.yml"
  tag "RGaSP"
  
  input:
    path tensors

  output:
    path "results_rgasp", emit: rgasp
    
  script:
  """
  # Execute R script for RGaSP evaluation
  echo "R version:"
  Rscript --version
  Rscript ${workflow.launchDir}/scripts/evaluate_rgasp.R \\
    --input-dir ${tensors} \\
    --output-dir results_rgasp
  """
} 