import argparse
import os
import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def viz_rmse_vs_train_time(benchmark_metrics, output_dir):
    names = list(benchmark_metrics.keys())
    rmses = [benchmark_metrics[name]["rmse"] for name in names]
    train_times = [benchmark_metrics[name]["train_time"] for name in names]
    plt.figure()
    tab10 = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "X", "h", "*", "p", "v", ">", "<"]
    for i, name in enumerate(names):
        color = tab10[i % len(tab10)]
        marker = markers[i % len(markers)]
        plt.scatter(train_times[i], rmses[i], color=color, marker=marker, s=70, label=name, edgecolors="black", linewidths=0.6, zorder=3)
    plt.xlabel("Training time (sec)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Training Time")
    plt.grid(True, linestyle=":", linewidth=0.6, zorder=0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_vs_training_time.png"))

def main():
    # 1. Parse arguments 
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-paths", required=True, type=ast.literal_eval,
                   help="A list of directory paths containing metrics.json files to benchmark")
    p.add_argument("--output-dir", required=True)
    
    args = p.parse_args()

    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Load metrics from each path
    if not isinstance(args.metrics_paths, list) or len(args.metrics_paths) == 0:
        raise ValueError("[benchmark_metrics] Error: --metrics-paths must be a non-empty list of directory paths")
    
    benchmark_metrics = {}
    for i, p in enumerate(args.metrics_paths):
        if not isinstance(p, str):
            raise ValueError(f"[benchmark_metrics] Error: item {i} must be a string path, got {type(p)}")
        metrics_path = os.path.join(p, "metrics.json")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"[benchmark_metrics] Missing metrics file for item {i}: {metrics_path}")
        with open(metrics_path, "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"[benchmark_metrics] Error decoding JSON for item {i}: {metrics_path}: {e}")
        model_name = metrics.get("name")
        if model_name is None:
            raise ValueError(f"[benchmark_metrics] Error: item {i} missing 'name' field")
        benchmark_metrics[model_name] = metrics
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
    
    # 6. Visualize RMSE vs Training Time scatter plot
    viz_rmse_vs_train_time(benchmark_metrics, args.output_dir)
    print("[benchmark_metrics] Generated RMSE vs Training Time scatter plot")

if __name__=="__main__":
    main()
