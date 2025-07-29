process evaluate_rgasp {
  tag "RGaSP"
  conda "environment.yml"
  
  input:
    path tensors

  output:
    path "results_rgasp", emit: rgasp
    
  script:
  """
  # Ensure conda environment is properly activated for R
  if [[ -n "\$CONDA_PREFIX" ]]; then
    export R_LIBS_USER="\$CONDA_PREFIX/lib/R/library"
    export R_LIBS="\$CONDA_PREFIX/lib/R/library"
    export R_LIBS_SITE="\$CONDA_PREFIX/lib/R/library"
    export PATH="\$CONDA_PREFIX/bin:\$PATH"
  fi
  
  # Execute R script for RGaSP evaluation
  echo "PATH: \$PATH"
  \$CONDA_PREFIX/bin/Rscript --version
  \$CONDA_PREFIX/bin/Rscript ${workflow.launchDir}/scripts/evaluate_rgasp.R \\
    --input-dir ${tensors} \\
    --output-dir results_rgasp
  """
} 