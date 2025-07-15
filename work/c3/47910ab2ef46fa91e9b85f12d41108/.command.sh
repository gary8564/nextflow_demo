#!/bin/bash -ue
# 1) If we're doing the tsunami case, grab the archive:
if [ "synthetic" == "tsunami_tokushima" ]; then
  echo '→ Downloading tsunami data…'
  gdown 'https://drive.google.com/file/d/1Sv9bdAyw3iHFvPd0vGHtj20Io_imt-2D/view?usp=sharing' \
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

python3 /Users/garychang/Documents/RWTH/HiWiJob/MBD/nextflow_demo/scripts/data_setup.py \
  --case-study synthetic \
   \
  --output-dir raw_data
