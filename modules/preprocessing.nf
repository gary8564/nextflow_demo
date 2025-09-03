process preprocessing {
  conda "${workflow.launchDir}/envs/preprocessing.yml"
  tag "preprocessing"
  input:
    path processed_data

  output:
    path "data_tensors", emit: tensors


 
  script:
  """
  # macOS-specific environment variable to avoid OpenMP error
  [[ "\$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE

  python ${workflow.launchDir}/scripts/preprocessing.py \
    --input-dir ${processed_data} \
    --output-dir data_tensors
  """
}
