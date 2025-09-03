process data_setup_zenodo {
  conda "${workflow.launchDir}/envs/data_setup_tsunami.yml"
  tag "${caseStudy}"
  
  input:
    tuple val(caseStudy), path(raw_data)
  
  output:
    path "processed_data", emit: processed

  script:
  """
  echo "[data_setup_zenodo] Processing Zenodo data for ${caseStudy}"
  
  # Process downloaded tsunami data
  args="--input-dir ${raw_data} --output-dir processed_data"
  echo "Running: python ${workflow.launchDir}/scripts/data_setup_tsunami.py \$args"
  python ${workflow.launchDir}/scripts/data_setup_tsunami.py \$args
  
  echo "[data_setup_zenodo] Successfully processed Zenodo data for ${caseStudy}"
  """
}
