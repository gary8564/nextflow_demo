# NextFlow Demo: GPytorch High-dimensional Input Problem (DKL)

A modular NextFlow pipeline for benchmarking Gaussian Process models on high-dimensional input problems.

## Workflow
![workflow](img/workflow.png)

The pipeline follows a 5-step workflow:
1. **Fetch Data**: Download or generate datasets based on configuration
2. **Data Setup**: Convert raw data to X/Y format using case-specific processors
3. **Preprocessing**: Standardize, split, and tensorize data for training
4. **Model Evaluation**: Train and evaluate both Exact GP and DKL models in parallel
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
│   └── benchmark_metrics.nf # Performance comparison and reporting
└── scripts/                 # Python implementation scripts
    ├── data_setup_synthetic.py  # Synthetic data generation
    ├── data_setup_tsunami.py    # Tsunami data processing
    ├── preprocessing.py         # Data preprocessing utilities
    ├── evaluate_exactgp.py      # Exact GP implementation
    ├── evaluate_dkl.py          # DKL implementation
    ├── benchmark_metrics.py     # Metrics calculation and comparison
    └── high_dim_input_prob.py   # Core GP/DKL model definitions
```

## Prerequisites
1. **Python >= 3.10**
2. **Conda Environment Manager**
   - [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) or 
   - [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
3. **NextFlow**
   - Follow the [installation instructions](https://www.nextflow.io/docs/latest/install.html)

## Usage
For synthetic case study:
```bash
nextflow run workflows/main.nf \
  --caseStudy synthetic \
  --outDir results \
  -profile local
```
 
For Tokushima Tsunami:
```bash
nextflow run workflows/main.nf \
  --caseStudy tsunami_tokushima \
  --outDir results \
  -profile local
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
