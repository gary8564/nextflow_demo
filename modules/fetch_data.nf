process fetch_data {
  tag "${caseStudy}"
  
  input:
    val caseStudy
  
  output:
    tuple val(caseStudy), path("raw_data/${caseStudy}"), emit: raw
  
  script:
  def dataset_config = params.datasets[caseStudy]
  def dataset_type = dataset_config.type
  
  if (dataset_type == "generate")
    """
    # Prepare synthetic data configuration
    mkdir -p raw_data/${caseStudy}
    
    # Store configuration parameters in metadata file
    cat > raw_data/${caseStudy}/.data_info << 'EOF'
type=${dataset_type}
description=${dataset_config.description}
${dataset_config.parameters.n_samples != null ? "n_samples=${dataset_config.parameters.n_samples}" : ""}
${dataset_config.parameters.seed != null ? "seed=${dataset_config.parameters.seed}" : ""}
EOF
    echo "[fetch_data] completed!"
    """
    
  else if (dataset_type == "zenodo")
    """
    # Download data from Zenodo
    mkdir -p raw_data/${caseStudy}
    cd raw_data/${caseStudy}
    
    echo "[fetch_data] Downloading ${caseStudy} from Zenodo (DOI: ${dataset_config.doi})"
    
    # Download files specified in conf/dataset.config
    ${dataset_config.files.collect { filename ->
      """
      echo "Downloading ${filename}..."
      wget -q --show-progress -O "${filename}" "${dataset_config.base_url}/files/${filename}?download=1"
      """
    }.join('\n    ')}
    
    # Extract all zip files found
    for zipfile in *.zip; do
      if [ -f "\$zipfile" ]; then
        echo "Unzipping \$zipfile..."
        unzip -q "\$zipfile"
        rm "\$zipfile"
      fi
    done
    
    # Create metadata file
    cat > .data_info << 'EOF'
type=${dataset_type}
description=${dataset_config.description}
doi=${dataset_config.doi}
source=${dataset_config.base_url}
EOF
    
    echo "[fetch_data] completed!"
    """
    
  else
    error "NotSupportedError: ${dataset_type} for ${caseStudy} is not supported!"
} 