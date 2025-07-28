import argparse
import os
import json
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

def main():
    # Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--exact-metrics-dir", required=True)
    p.add_argument("--dkl-metrics-dir",   required=True)
    p.add_argument("--output-dir",    required=True)
    args = p.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    exact = json.load(open(os.path.join(args.exact_metrics_dir, "metrics.json")))
    dkl   = json.load(open(os.path.join(args.dkl_metrics_dir, "metrics.json")))
    
    # Visualize predictions and save to output_dir
    predictions_exact = np.concatenate([np.array(exact["predictions_mean"])[:,np.newaxis], 
                                        np.array(exact["predictions_std"])[:,np.newaxis]], 
                                       axis=1)
    predictions_dkl = np.concatenate([np.array(dkl["predictions_mean"])[:,np.newaxis], 
                                      np.array(dkl["predictions_std"])[:,np.newaxis]], 
                                     axis=1)
    viz_prediction(np.array(exact["ground_truth"]), predictions_exact, "ExactGP", args.output_dir)
    viz_prediction(np.array(dkl["ground_truth"]), predictions_dkl, "DKL", args.output_dir)

    # Save evaluation metrics to csv
    benchmark_metric_exact = dict(rmse=exact["rmse"], train_time=exact["train_time"], infer_time=exact["infer_time"])
    benchmark_metric_dkl = dict(rmse=dkl["rmse"], train_time=dkl["train_time"], infer_time=dkl["infer_time"])
    df = pd.DataFrame([benchmark_metric_exact,benchmark_metric_dkl], index=["ExactGP", "DKL"])
    csv = os.path.join(args.output_dir,"comparison.csv")
    df.to_csv(csv)
    print(f"[benchmark_metrics] saved → {csv}")

if __name__=="__main__":
    main()
