process data_setup {
  tag "${params.caseStudy}"
  input:
    val caseStudy
  output:
    path "raw_data", emit: raw

  script:
  """
  # 1) If we're doing the tsunami case, grab the archive:
  if [ \"${caseStudy}\" == \"tsunami_tokushima\" ]; then
    echo '→ Downloading tsunami data…'
    gdown 'https://drive.google.com/file/d/1Sv9bdAyw3iHFvPd0vGHtj20Io_imt-2D/view?usp=sharing' \\
      -O tsunami.zip
    echo '→ Unzipping tsunami.zip…'
    unzip -q tsunami.zip -d tsunami_data
    rm tsunami.zip
    INPUT_DIR=tsunami_data
  else
    INPUT_DIR=
  fi

  # 2) Run your Python data_setup script, pointing it at
  #    either the unzipped tsunami_data/ or letting it
  #    generate the synthetic data.

  python3 ${workflow.launchDir}/scripts/data_setup.py \\
    --case-study ${caseStudy} \\
    ${ caseStudy == 'tsunami_tokushima' ? "--input-dir \${INPUT_DIR}" : "" } \\
    --output-dir raw_data
  """
}