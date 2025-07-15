# NextFlow Demo: GPytorch High-dimensional Input Problem (DKL)

## Workflow
![workflow](img/workflow.png)

## Folder structure
```
.
├── nextflow.config
├── environment.yml
├── requirements.txt
├── README.md
├── img
│   └── workflow.png
├── workflows/
│   └── main.nf
├── modules/
│   ├── data_setup.nf
│   ├── preprocessing.nf
│   ├── train_exactgp.nf
│   ├── train_dkl.nf
│   └── evaluate_metrics.nf
└── scripts/
    ├── data_setup.py
    ├── preprocessing.py
    ├── train_exactgp.py
    ├── train_dkl.py
    └── evaluate_metrics.py
```

## Prerequisites
1. Python >= 3.10
2. [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
3. NextFlow
Follow the [instructions](https://www.nextflow.io/docs/latest/install.html) to install NextFlow.

## Usage
```bash
nextflow run workflows/main.nf \
  --caseStudy synthetic \
  --outDir results \
  -profile local
```

