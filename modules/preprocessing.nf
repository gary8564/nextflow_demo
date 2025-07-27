process preprocessing {
  tag "preprocessing"
  input:
    path processed_data

  output:
    path "data_tensors", emit: tensors

  script:
  """
  python3 ${workflow.launchDir}/scripts/preprocessing.py \
    --input-dir ${processed_data} \
    --output-dir data_tensors
  """
}
