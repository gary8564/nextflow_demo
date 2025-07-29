import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def viz_prediction(ground_truth, predictions, model_name, output_dir):
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.reshape(-1, 1)
    if predictions.ndim == 2:
        prediction_mean = predictions[:, 0]
        prediction_std = predictions[:, 1]
    elif predictions.ndim == 3:
        prediction_mean = predictions[:, :, 0]
        prediction_std = predictions[:, :, 1]
    else:
        raise ValueError(f"The dimension of predictions must be 2d or 3d np.ndarray, but got {predictions.ndim}.")
    title = f"Prediction v.s. Ground-truth for {model_name} Model"
    plt.figure()
    y_true = ground_truth.flatten()
    y_pred = prediction_mean.flatten()
    plt.plot([np.min(y_true),np.max(y_true)], [np.min(y_true),np.max(y_true)], color='black')
    pred_std = prediction_std.flatten()
    plt.errorbar(y_true, y_pred, 
                    yerr=pred_std, 
                    fmt='o', 
                    markersize=6,
                    markeredgewidth=1.0,
                    markerfacecolor='None',
                    ecolor='cornflowerblue',
                    elinewidth=0.5,
                    capsize=2,
                    alpha=0.8,  
                    label='emulator prediction ± std',
                    zorder=1)
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.xlim(np.min(y_true),np.max(y_true))
    plt.ylim(np.min(y_true),np.max(y_true))
    plt.legend()
    plt.tight_layout()
    plt.title(title)
    plt.savefig(os.path.join(output_dir, f"{model_name}.png"))

def main():
    # 1. Parse arguments 
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    # Add optional method-specific arguments
    p.add_argument("--exactgp-metrics-dir", required=False)
    p.add_argument("--dkl-metrics-dir", required=False)
    p.add_argument("--rgasp-metrics-dir", required=False)
    
    args = p.parse_args()

    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Load metrics for all selected models ready to be benchmarked
    model_configs = {
        'ExactGP': args.exactgp_metrics_dir,
        'DKL': args.dkl_metrics_dir,
        'RGaSP': args.rgasp_metrics_dir,
    }
    benchmark_models = {name: path for name, path in model_configs.items() 
                     if path is not None and os.path.exists(os.path.join(path, "metrics.json"))}
    if not benchmark_models:
        print("Error: No valid model result directories found!")
        sys.exit(1)
    print(f"[benchmark_metrics] Processing {len(benchmark_models)} models: {list(benchmark_models.keys())}")
    benchmark_metrics = {}
    for model_name, metrics_dir in benchmark_models.items():
        metrics_file = os.path.join(metrics_dir, "metrics.json")
        benchmark_metrics[model_name] = json.load(open(metrics_file))
        print(f"[benchmark_metrics] Loaded {model_name} metrics from {metrics_file}")
    
    # 4. Visualize prediction for each model
    for model_name, metrics in benchmark_metrics.items():
        predictions = np.concatenate([
            np.array(metrics["predictions_mean"])[:,np.newaxis], 
            np.array(metrics["predictions_std"])[:,np.newaxis]
        ], axis=1)
        viz_prediction(np.array(metrics["ground_truth"]), predictions, model_name, args.output_dir)
        print(f"[benchmark_metrics] Generated visualization for {model_name}")

    # 5. Save evaluation metrics comparison as a csv 
    benchmark_data = []
    model_names = []
    for model_name, metrics in benchmark_metrics.items():
        benchmark_row = {
            'rmse': metrics["rmse"], 
            'train_time': metrics["train_time"], 
            'infer_time': metrics["infer_time"]
        }
        benchmark_data.append(benchmark_row)
        model_names.append(model_name)
    df = pd.DataFrame(benchmark_data, index=model_names)
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    df.to_csv(csv_path)
    print(f"[benchmark_metrics] Saved comparison → {csv_path}")

if __name__=="__main__":
    main()
