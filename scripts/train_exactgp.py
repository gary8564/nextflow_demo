import argparse
import os
import json
import torch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from gpytorch_emulator import ExactGP
from gpytorch_emulator.utils import ErrorMetrics

def descale_data(test_y, mean_scaled, std_scaled, lower95_scaled, upper95_scaled, standardizer: StandardScaler):
    mu = standardizer.mean_[-1] 
    sigma = standardizer.scale_[-1] 
    groud_truth = test_y * sigma + mu
    mean = mean_scaled * sigma + mu
    std = std_scaled * sigma 
    lower95 = lower95_scaled * sigma + mu
    upper95 = upper95_scaled * sigma + mu
    return groud_truth, mean, std, lower95, upper95

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    X_train = torch.load(os.path.join(args.input_dir,"train_X.pt")).numpy()
    y_train = torch.load(os.path.join(args.input_dir,"train_y.pt")).numpy()
    X_test = torch.load(os.path.join(args.input_dir,"test_X.pt")).numpy()
    y_test = torch.load(os.path.join(args.input_dir,"test_y.pt")).numpy()
    
    scaler = joblib.load(os.path.join(args.input_dir,"scaler.pkl"))

    model = ExactGP(device="cuda" if torch.cuda.is_available() else "cpu",
                    kernel_type="matern_5_2")
    training_time = model.train(X_train, y_train)
    mean_scaled, std_scaled, lower95_scaled, upper95_scaled, infer_time  = model.predict(X_test)

    ground_truth, mean, std, lower95, upper95 = descale_data(
        y_test, mean_scaled, std_scaled, lower95_scaled, upper95_scaled, scaler)

    rmse = ErrorMetrics.RMSE(mean, ground_truth)
    metrics = dict(rmse=rmse, train_time=training_time, infer_time=infer_time)

    with open(os.path.join(args.output_dir,"metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[ExactGP] metrics → {args.output_dir}/metrics.json")

if __name__=="__main__":
    main()