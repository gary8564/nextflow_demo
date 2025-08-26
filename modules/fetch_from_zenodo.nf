process fetch_from_zenodo {
  tag "${caseStudy}"
  
  input:
    val caseStudy
  
  output:
    tuple val(caseStudy), path("raw_data/${caseStudy}"), emit: raw
  
  script:
  def dataset_config = params.datasets[caseStudy]
  
  """
  # Download data from Zenodo
  mkdir -p raw_data/${caseStudy}
  cd raw_data/${caseStudy}
  
  echo "[fetch_from_zenodo] Downloading ${caseStudy} from Zenodo (DOI: ${dataset_config.doi})"
  
  # Download files specified in conf/datasets.config
  ${dataset_config.files.collect { filename ->
    """
    echo "Downloading ${filename}..."
    wget -q --show-progress -O "${filename}" "${dataset_config.base_url}/files/${filename}?download=1"
    """
  }.join('\n  ')}
  
  # Extract all zip files found
  for zipfile in *.zip; do
    if [ -f "\$zipfile" ]; then
      echo "Unzipping \$zipfile..."
      unzip -q "\$zipfile"
      rm "\$zipfile"
    fi
  done
  
  echo "[fetch_from_zenodo] completed!"
  """
}
