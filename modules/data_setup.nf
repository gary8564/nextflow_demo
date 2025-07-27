process data_setup {
  tag "${caseStudy}"
  
  input:
    tuple val(caseStudy), path(raw_data)
  
  output:
    path "processed_data", emit: processed

  script:
  """
  # Read dataset type and configuration from .data_info
  dataset_type=\$(grep "^type=" ${raw_data}/.data_info | cut -d'=' -f2)
  
  echo "[data_setup] Processing \$dataset_type data for ${caseStudy}"
  
  if [ "\$dataset_type" = "generate" ]; then
    # Generate synthetic data using configuration from .data_info with defaults
    echo "Generating synthetic data..."
    
    # Build arguments - only add parameters that are configured
    args="--output-dir processed_data"
    
    # Add n_samples if specified in config
    n_samples=\$(grep "^n_samples=" ${raw_data}/.data_info 2>/dev/null | cut -d'=' -f2)
    if [ -n "\$n_samples" ]; then
      args="\$args --n-samples \$n_samples"
    fi
    
    # Add seed if specified in config
    seed=\$(grep "^seed=" ${raw_data}/.data_info 2>/dev/null | cut -d'=' -f2)
    if [ -n "\$seed" ]; then
      args="\$args --seed \$seed"
    fi
    
    python3 ${workflow.launchDir}/scripts/data_setup_synthetic.py \$args
    
  elif [ "\$dataset_type" = "zenodo" ]; then
    # Process downloaded tsunami data
    echo "Processing downloaded tsunami data..."
    
    args="--input-dir ${raw_data} --output-dir processed_data"
    python3 ${workflow.launchDir}/scripts/data_setup_tsunami.py \$args
      
  else
    echo "ERROR: Unknown dataset type: \$dataset_type"
    exit 1
  fi

  
  echo "[data_setup] Successfully processed ${caseStudy} data"
  """
}