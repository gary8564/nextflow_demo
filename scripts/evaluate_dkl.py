import os
import torch
import numpy as np
import argparse
import json
import h5py
from high_dim_gp.emulator import DKL_GP
from high_dim_gp.utils import ErrorMetrics

def descale_data(test_y: np.ndarray, 
                 mean_scaled: np.ndarray, 
                 std_scaled: np.ndarray, 
                 lower95_scaled: np.ndarray, 
                 upper95_scaled: np.ndarray, 
                 scaler_mean: np.ndarray,
                 scaler_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Descale the data to the original scale
    
    Args:
        test_y: The scaled test data
        mean_scaled: The mean of the scaled test data
        std_scaled: The standard deviation of the scaled test data
        lower95_scaled: The lower 95% confidence interval of the scaled test data
        upper95_scaled: The upper 95% confidence interval of the scaled test data
        scaler_mean: Mean values from standardization
        scaler_scale: Scale values from standardization
        
    Returns:
        groud_truth: The original test data
        mean: The mean of the original test data
        std: The standard deviation of the original test data
        lower95: The lower 95% confidence interval of the original test data
        upper95: The upper 95% confidence interval of the original test data
    """
    mu = scaler_mean[-1] 
    sigma = scaler_scale[-1] 
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
    
    # 2. Load data from HDF5
    hdf5_file = os.path.join(args.input_dir, "data.h5")
    with h5py.File(hdf5_file, 'r') as f:
        X_train = torch.from_numpy(f['train_X'][:]).float()
        y_train = torch.from_numpy(f['train_y'][:]).float()
        X_test = torch.from_numpy(f['test_X'][:]).float()
        y_test = torch.from_numpy(f['test_y'][:]).float()
        
        # Load standardization parameters
        scaler_mean = f['scaler_mean'][:]
        scaler_scale = f['scaler_scale'][:]
    
    # 3. Train
    model = DKL_GP(reduced_dim=2,
                   device=args.device,
                   kernel_type="matern_5_2")
    training_time = model.train(X_train, y_train)
    
    # 4. Predict
    mean_scaled, std_scaled, lower95_scaled, upper95_scaled, infer_time = model.predict(X_test)
    
    # 5. Descale 
    ground_truth, mean, std, lower95, upper95 = descale_data(y_test.detach().cpu().numpy(), mean_scaled, std_scaled, lower95_scaled, upper95_scaled, scaler_mean, scaler_scale)
    
    # 6. Evaluate metrics
    rmse = ErrorMetrics.RMSE(mean, ground_truth)

    # 7. Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = dict(
        name="DKL",
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
    print(f"[DKL] metrics → {args.output_dir}/metrics.json")
    
if __name__=="__main__":
    main()
