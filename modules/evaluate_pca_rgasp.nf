process evaluate_pca_rgasp {
  tag "PCA-RGaSP"

  input:
    path tensors

  output:
    path "results_pca_rgasp", emit: pca_rgasp

  script:
  """
  # macOS-specific environment variable to avoid OpenMP error
  [[ "\$(uname)" == "Darwin" ]] && export KMP_DUPLICATE_LIB_OK=TRUE

  python ${workflow.launchDir}/scripts/evaluate_pca_rgasp.py \
    --input-dir ${tensors} \
    --output-dir results_pca_rgasp \
    ${params.useGPU ? '--device gpu' : '--device cpu'}
  """
}


