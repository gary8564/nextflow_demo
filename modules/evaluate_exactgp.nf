process evaluate_exactgp {
  tag "ExactGP"
  accelerator 1 

  input:
    path tensors

  output:
    path "results_exactgp", emit: exactgp

  script:
  """
  # macOS-specific environment variable to avoid OpenMP error
  [[ "\$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE
  
  python ${workflow.launchDir}/scripts/evaluate_exactgp.py \
    --input-dir  ${tensors} \
    --output-dir results_exactgp \
    ${params.useGPU ? '--device cuda' : '--device cpu'}
  """
}
