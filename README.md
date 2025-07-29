# NextFlow Demo: GPytorch High-dimensional Input Problem (DKL)

A **language-agnostic** modular NextFlow pipeline for benchmarking Gaussian Process models on high-dimensional input problems. Supports Python (GPyTorch) and R (RobustGaSP) implementations with HDF5 data interchange.

## TODO

### Incomplete
- [ ] Language-agnostic data interchange
  - [ ] **PCA-RGaSP** not implemented.
### Done 

- [x] Download from sources
- [x] Working workflow
  - [x] Separation of data setup and downloading
  - [x] Using conditional scripts to separate different dataset case studies
- [x] GPytorch training hardware options: cpu/gpu
- [x] Add plotting functions in benchmark metrics
- [x] Renaming and documenting the code in clearer way
- [x] Language-agnostic data interchange
  - [x] **HDF5-based data format** for cross-language compatibility
  - [x] **RGaSP evaluation in R** - Added RobustGaSP Gaussian Process implementation
  - [x] **Different GP models comparison** - ExactGP, DKL (Python), and RGaSP (R)

## Workflow
![workflow](img/workflow.png)

The pipeline follows a 5-step workflow:
1. **Fetch Data**: Download or generate datasets based on configuration
2. **Data Setup**: Convert raw data to X/Y format using case-specific processors
3. **Preprocessing**: Standardize, split, and save data in **HDF5 format** (language-agnostic)
4. **Model Evaluation**: Train and evaluate **three GP models in parallel**:
   - **ExactGP** (Python/GPyTorch)
   - **DKL** (Python/GPyTorch) 
   - **RGaSP** (R/RobustGaSP)
5. **Benchmark Metrics**: Compare model performance and save results

## Supported Datasets

### Synthetic Dataset
- **Type**: Generated high-dimensional synthetic data
- **Description**: 100D synthetic function for testing purposes
- **Configuration**: Customizable sample size and random seed for reproducibility

### Tsunami Dataset
- **Type**: Real-world data from Zenodo repository
- **Description**: Neural network-based surrogate model for tsunami inundation assessment
- **Source**: [DOI: 10.5281/zenodo.15093228](https://zenodo.org/records/15093228)
- **Data**: Initial water level distributions, tsunami water level distributions, and inundation distributions

## Folder Structure
```
.
├── nextflow.config          # Main NextFlow configuration
├── environment.yml          # Conda environment specification
├── requirements.txt         # Python dependencies
├── README.md
├── img/
│   └── workflow.png         # Pipeline workflow diagram
├── conf/                    # Configuration files
│   ├── conda.config         # Conda-specific settings
│   ├── datasets.config      # Dataset definitions and parameters
│   └── profiles.config      # Execution profiles (local, HPC, etc.)
├── workflows/
│   └── main.nf              # Main workflow orchestration
├── modules/                 # Modular NextFlow processes
│   ├── fetch_data.nf        # Data fetching (download/generate)
│   ├── data_setup.nf        # Data processing coordinator
│   ├── preprocessing.nf     # Data standardization and splitting
│   ├── evaluate_exactgp.nf  # Exact GP model training and evaluation
│   ├── evaluate_dkl.nf      # DKL model training and evaluation
│   ├── evaluate_rgasp.nf    # RGaSP model training and evaluation
│   └── benchmark_metrics.nf # Performance comparison and reporting
├── scripts/                 # Implementation scripts 
│   ├── data_setup_synthetic.py  # Synthetic data generation
│   ├── data_setup_tsunami.py    # Tsunami data processing
│   ├── preprocessing.py         # Data preprocessing utilities
│   ├── evaluate_exactgp.py      # Exact GP implementation (Python/GPyTorch)
│   ├── evaluate_dkl.py          # DKL implementation (Python/GPyTorch)
│   ├── evaluate_rgasp.R         # RGaSP implementation (R/RobustGaSP)
│   └── benchmark_metrics.py     # Metrics calculation and comparison
└── results/                 # Pipeline outputs
    └── benchmark_results/   # Model comparison results
        ├── comparison.csv   # Performance metrics comparison
        ├── ExactGP.png      # Exact GP performance plots
        ├── DKL.png          # DKL performance plots
        └── RGaSP.png        # RGaSP performance plots
```

## Prerequisites
1. **Python >= 3.10**
2. **Conda Environment Manager**
   - [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) or 
   - [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
3. **NextFlow**
   - Follow the [installation instructions](https://www.nextflow.io/docs/latest/install.html)

## Basic Usage
For synthetic case study:
```bash
nextflow run workflows/main.nf \
  --caseStudy synthetic \
  --outDir results \
  -profile local
```

For synthetic case study with GPU acceleration:
```bash
nextflow run workflows/main.nf \
  --caseStudy synthetic \
  --outDir results \
  --useGPU true \
  -profile local
```
 
For Tokushima Tsunami:
```bash
nextflow run workflows/main.nf \
  --caseStudy tsunami_tokushima \
  --outDir results \
  -profile local
```

For Tokushima Tsunami with GPU acceleration:
```bash
nextflow run workflows/main.nf \
  --caseStudy tsunami_tokushima \
  --outDir results \
  --useGPU true \
  -profile local
```

## Advanced Usage
### Workflow DAG generation
```bash
nextflow run workflows/main.nf \
  --caseStudy tsunami_tokushima \
  --outDir results \
  -profile local \
  -with-dag flowchart.png
```

### Configuration Options
1. **Dataset Configuration**
Extensibility for new dataset study case through the `conf/datasets.config` file
```groovy
params {
  datasets = [
    // ... existing datasets ...
    
    my_new_study: [
      type: "zenodo",  // or "generate" for synthetic data
      description: "Description of your dataset",
      doi: "10.5281/zenodo.XXXXXXX",  // for Zenodo datasets
      base_url: "https://zenodo.org/records/XXXXXXX",  // for Zenodo datasets
      files: [  // for Zenodo datasets
        "data_file1.csv",
        "data_file2.zip"
      ],
      parameters: [  // for generated datasets
        n_samples: 1000,
        custom_param: "value"
      ]
    ]
  ]
}
```

2. **Execution Profiles Configuration**
- `local`: Run on local machine
- Add custom profiles in `conf/profiles.config` for HPC clusters

### Steps to Add a New Case Study
#### 1. Add Dataset Configuration
Edit `conf/datasets.config` to define your new case study:


#### 2. Create Data Processing Script
Create a new Python script in the `scripts/` directory:

- **For Zenodo datasets**: `scripts/data_setup_mynewstudy.py`
- **For generated datasets**: `scripts/data_setup_mynewstudy.py`

The script should:
- Accept `--input-dir` (for downloaded data) or generation parameters
- Accept `--output-dir` for processed data
- Output standardized `X.npy` and `Y.npy` files

**Example script structure:**
```python
import argparse
import numpy as np
import os

def process_data(input_dir, output_dir):
    """Process raw data into X, Y format"""
    # Your data processing logic here
    X = ...  # Features as numpy array
    Y = ...  # Targets as numpy array
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/X.npy", X)
    np.save(f"{output_dir}/Y.npy", Y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="Input directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    # Add your custom parameters here
    args = parser.parse_args()
    
    process_data(args.input_dir, args.output_dir)
```

#### 3. Update Data Setup Module (if needed)
If your data source type is not `zenodo` or `generate`, extend `modules/data_setup.nf`:

```bash
elif [ "$dataset_type" = "my_new_type" ]; then
  echo "Processing my new data type..."
  args="--input-dir ${raw_data} --output-dir processed_data"
  python ${workflow.launchDir}/scripts/data_setup_mynewstudy.py $args
```

#### 4. Extend Fetch Data Module (for new source types)
For new data source types, extend `modules/fetch_data.nf`:

```bash
else if (dataset_type == "my_new_source")
  """
  # Your custom data fetching logic
  mkdir -p raw_data/${caseStudy}
  # ... fetching implementation ...
  """
```

#### 5. Run the pipeline:
```bash
nextflow run workflows/main.nf --caseStudy my_csv_study --outDir results -profile local
```

### Steps to Add a New GP Model 

To add a new GP implementation (e.g., R, MATLAB, Julia, and so on):

#### 1. Create Evaluation Script
Create a new script in the `scripts/` directory (e.g., `scripts/evaluate_newmethod.py` or `scripts/evaluate_newmethod.R`):

**Example in R:**
```r
#!/usr/bin/env Rscript
library(hdf5r)
library(jsonlite)
library(optparse)

# Parse arguments
option_list <- list(
  make_option(c("--input-dir"), type="character", default=NULL,
              help="Input directory containing data.h5"),
  make_option(c("--output-dir"), type="character", default=NULL,
              help="Output directory for metrics.json")
)
opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Load data and implement your GP method
# ... your implementation ...

# Save metrics
dir.create(opt$`output-dir`, recursive=TRUE, showWarnings=FALSE)
write_json(metrics, file.path(opt$`output-dir`, "metrics.json"))
```

#### 2. Create Nextflow Module
Create a new module file `modules/evaluate_newmethod.nf`:

```groovy
process evaluate_newmethod {
  tag "NewMethod"
  conda "environment.yml"
  
  input:
    path tensors

  output:
    path "results_newmethod", emit: newmethod
    
  script:
  """
  # For Python implementations
  python ${workflow.launchDir}/scripts/evaluate_newmethod.py \\
    --input-dir ${tensors} \\
    --output-dir results_newmethod
  
  # For R implementations
  # Rscript ${workflow.launchDir}/scripts/evaluate_newmethod.R \\
  #   --input-dir ${tensors} \\
  #   --output-dir results_newmethod
  """
}
```

#### 3. Update Main Workflow
Add your new method to `workflows/main.nf`:

```groovy
#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { fetch_data       } from '../modules/fetch_data.nf'
include { data_setup       } from '../modules/data_setup.nf'
include { preprocessing    } from '../modules/preprocessing.nf'
include { evaluate_exactgp } from '../modules/evaluate_exactgp.nf'
include { evaluate_dkl     } from '../modules/evaluate_dkl.nf'
include { evaluate_rgasp   } from '../modules/evaluate_rgasp.nf'
include { evaluate_newmethod } from '../modules/evaluate_newmethod.nf'  // Add this line
include { benchmark_metrics } from '../modules/benchmark_metrics.nf'

workflow {
    println "▶ Starting pipeline with caseStudy=${params.caseStudy}"
    
    // 1. Fetch raw inputs (download or generate metadata)
    def raw_ch = fetch_data( params.caseStudy )
    // 2. Convert raw -> X/Y
    def data_ch = data_setup( raw_ch )
    // 3. Standardize, split, save to HDF5
    def tensors_ch = preprocessing( data_ch )
    // 4. Train emulators in parallel
    def exactgp_ch = evaluate_exactgp( tensors_ch )
    def dkl_ch     = evaluate_dkl( tensors_ch )
    def rgasp_ch   = evaluate_rgasp( tensors_ch )
    def newmethod_ch = evaluate_newmethod( tensors_ch )  // Add this line
    // 5. Compare metrics and save results
    benchmark_metrics( exactgp_ch, dkl_ch, rgasp_ch, newmethod_ch )  // Add newmethod_ch
}
```

#### 4. Update Benchmark Metrics Module
Modify `modules/benchmark_metrics.nf` to include your new method:

```groovy
process benchmark_metrics {
  tag "benchmark_metrics"
  publishDir "results", mode: 'copy'
  
  input:
    path exactgp_results
    path dkl_results
    path rgasp_results
    path newmethod_results  // Add this line

  output:
    path "benchmark_results", emit: result_dir

  script:
  """
  python ${workflow.launchDir}/scripts/benchmark_metrics.py \\
    --exactgp-metrics-dir ${exactgp_results} \\
    --dkl-metrics-dir ${dkl_results} \\
    --rgasp-metrics-dir ${rgasp_results} \\
    --newmethod-metrics-dir ${newmethod_results} \\  # Add this line
    --output-dir benchmark_results
  """
}
```

#### 5. Update Benchmark Metrics Script
Modify `scripts/benchmark_metrics.py` to handle your new method's results and include it in comparisons.

#### 6. Update Environment (if needed)
If your new method requires additional dependencies, update `environment.yml` or `requirements.txt`.

#### 7. Test Your Implementation
Run the pipeline with your new method:
```bash
nextflow run workflows/main.nf --caseStudy synthetic --outDir results -profile local
```

## Language-Agnostic Design

This workflow demonstrates **programming language agnosticism** in scientific computing pipelines:

### Data Interchange Format
- **HDF5**: Cross-language scientific data format
  - Stores numerical arrays natively (no serialization overhead)
  - Hierarchical structure for organized data (train/test splits)
  - Metadata support for standardization parameters

## Supported GP Implementations

| Model | Language | Library | Kernel | Features |
|-------|----------|---------|---------|----------|
| **ExactGP** | Python | GPyTorch | Matérn 5/2 | Exact inference, GPU support |
| **DKL** | Python | GPyTorch | Matérn 5/2 | Deep kernel learning, GPU support |
| **RGaSP** | R | RobustGaSP | Matérn 5/2 | Robust Gaussian processes, outlier handling |
