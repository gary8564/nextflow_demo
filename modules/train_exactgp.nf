process train_exactgp {
  tag "ExactGP"
  input:
    path tensors

  output:
    path "results_exactgp", emit: exactgp

  script:
  """
  export KMP_DUPLICATE_LIB_OK=TRUE
  
  python3 ${workflow.launchDir}/scripts/train_exactgp.py \
    --input-dir  ${tensors} \
    --output-dir results_exactgp
  """
}
