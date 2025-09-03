process evaluate_dkl {
  tag "DKL"

  accelerator 2

  input:
    path tensors

  output:
    path "results_dkl", emit: dkl

  script:
  """
  # macOS-specific environment variable to avoid OpenMP error
  [[ "\$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE
  
  python ${workflow.launchDir}/scripts/evaluate_dkl.py \
    --input-dir ${tensors} \
    --output-dir results_dkl \
    ${params.useGPU ? '--device gpu' : '--device cpu'}
  """
}
