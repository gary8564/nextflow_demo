import argparse
import os
import json
import torch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from gpytorch_emulator import ExactGP
from gpytorch_emulator.utils import ErrorMetrics

def descale_data(test_y: np.ndarray,
                 mean_scaled: np.ndarray,
                 std_scaled: np.ndarray,
                 lower95_scaled: np.ndarray,
                 upper95_scaled: np.ndarray,
                 standardizer: StandardScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Descale the data to the original scale
    
    Args:
        test_y: The scaled test data
        mean_scaled: The mean of the scaled test data
        std_scaled: The standard deviation of the scaled test data
        lower95_scaled: The lower 95% confidence interval of the scaled test data
        upper95_scaled: The upper 95% confidence interval of the scaled test data
        standardizer: The standardizer used to scale the data
        
    Returns:
        groud_truth: The original test data
        mean: The mean of the original test data
        std: The standard deviation of the original test data
        lower95: The lower 95% confidence interval of the original test data
        upper95: The upper 95% confidence interval of the original test data
    """
    mu = standardizer.mean_[-1] 
    sigma = standardizer.scale_[-1] 
    groud_truth = test_y * sigma + mu
    mean = mean_scaled * sigma + mu
    std = std_scaled * sigma 
    lower95 = lower95_scaled * sigma + mu
    upper95 = upper95_scaled * sigma + mu
    return groud_truth, mean, std, lower95, upper95

def main():
    # 1. Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    
    # 2. Train
    X_train = torch.load(os.path.join(args.input_dir,"train_X.pt"))
    y_train = torch.load(os.path.join(args.input_dir,"train_y.pt"))
    model = ExactGP(device=args.device,
                    kernel_type="matern_5_2")
    training_time = model.train(X_train, y_train)
    
    # 3. Predict
    X_test = torch.load(os.path.join(args.input_dir,"test_X.pt"))
    y_test = torch.load(os.path.join(args.input_dir,"test_y.pt"))
    mean_scaled, std_scaled, lower95_scaled, upper95_scaled, infer_time  = model.predict(X_test)
    
    # 4. Descale
    scaler = joblib.load(os.path.join(args.input_dir,"scaler.pkl"))
    ground_truth, mean, std, lower95, upper95 = descale_data(
        y_test.detach().cpu().numpy(), mean_scaled, std_scaled, lower95_scaled, upper95_scaled, scaler)
    
    # 5. Evaluate metrics
    rmse = ErrorMetrics.RMSE(mean, ground_truth)
    
    # 6. Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = dict(
        ground_truth=ground_truth.tolist(),
        predictions_mean=mean.tolist(),
        predictions_std=std.tolist(),
        predictions_lower95=lower95.tolist(),
        predictions_upper95=upper95.tolist(),
        rmse=float(rmse),
        train_time=float(training_time),
        infer_time=float(infer_time)
    )
    with open(os.path.join(args.output_dir,"metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ExactGP] metrics → {args.output_dir}/metrics.json")

if __name__=="__main__":
    main()