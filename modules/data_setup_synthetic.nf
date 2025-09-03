process data_setup_synthetic {
  conda "${workflow.launchDir}/envs/data_setup_synthetic.yml"
  tag "${caseStudy}"
  
  input:
    val caseStudy
  
  output:
    path "processed_data", emit: processed

  script:
  def dataset_config = params.datasets[caseStudy]
  
  """
  echo "[data_setup_synthetic] Generating synthetic data for ${caseStudy}"
  
  # Build arguments - only add parameters that are configured
  args="--output-dir processed_data"
  
  # Add n_samples if specified in config
  ${dataset_config.parameters.n_samples != null ? """
  args="\$args --n-samples ${dataset_config.parameters.n_samples}"
  """ : ""}
  
  # Add seed if specified in config
  ${dataset_config.parameters.seed != null ? """
  args="\$args --seed ${dataset_config.parameters.seed}"
  """ : ""}
  
  echo "Running: python ${workflow.launchDir}/scripts/data_setup_synthetic.py \$args"
  python ${workflow.launchDir}/scripts/data_setup_synthetic.py \$args
  
  echo "[data_setup_synthetic] Successfully generated synthetic data for ${caseStudy}"
  """
}
