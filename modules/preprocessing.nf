process preprocessing {
  tag "preprocessing"
  input:
    path raw

  output:
    path "data_tensors", emit: tensors

  script:
  """
  python3 ${workflow.launchDir}/scripts/preprocessing.py \
    --input-dir ${raw} \
    --output-dir data_tensors
  """
}
