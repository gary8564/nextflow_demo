process train_dkl {
  tag "DKL"

  input:
    path tensors

  output:
    path "results_dkl", emit: dkl

  script:
  """
  export KMP_DUPLICATE_LIB_OK=TRUE
  
  python3 ${workflow.launchDir}/scripts/train_dkl.py \
    --input-dir ${tensors} \
    --output-dir results_dkl
  """
}
