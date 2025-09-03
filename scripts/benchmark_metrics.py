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
    p.add_argument("--metrics-file", required=True)
    p.add_argument("--output-dir", required=True)
    
    args = p.parse_args()

    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Load metrics list
    with open(args.metrics_file, "r") as f:
        metrics_list = json.load(f)
    if not isinstance(metrics_list, list) or len(metrics_list) == 0:
        raise ValueError("[benchmark_metrics] Error: --metrics-file must be a JSON array of metrics dicts")
    # Validate and index by model name
    benchmark_metrics = {}
    for i, m in enumerate(metrics_list):
        if not isinstance(m, dict):
            raise ValueError(f"[benchmark_metrics] Error: item {i} in metrics list is not a dict")
        model_name = m.get("name")
        if model_name is None:
            raise ValueError(f"[benchmark_metrics] Error: item {i} missing 'name' field")
        benchmark_metrics[model_name] = m
    print(f"[benchmark_metrics] Processing {len(benchmark_metrics)} models: {list(benchmark_metrics.keys())}")
    
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
