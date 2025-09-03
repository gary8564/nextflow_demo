import argparse
import os
import json
import numpy as np
import h5py
from psimpy.emulator import InputDimReducer, PCAScalarGaSP, LinearPCA
from gpytorch_emulator.utils import ErrorMetrics
from psimpy.utility.util_funcs import reduced_dim

def descale_data(test_y: np.ndarray,
                 mean_scaled: np.ndarray,
                 std_scaled: np.ndarray,
                 lower95_scaled: np.ndarray,
                 upper95_scaled: np.ndarray,
                 scaler_mean: np.ndarray,
                 scaler_scale: np.ndarray):
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
        X_train = f['train_X'][:]
        y_train = f['train_y'][:]
        X_test = f['test_X'][:]
        y_test = f['test_y'][:]
        
        # Load standardization parameters
        scaler_mean = f['scaler_mean'][:]
        scaler_scale = f['scaler_scale'][:]
    
    # 3. Train
    pca = LinearPCA()
    input_reducer  = InputDimReducer(pca)
    model = PCAScalarGaSP(
        ndim=int(reduced_dim(input_reducer, X_train)),
        input_dim_reducer=input_reducer,
    )
    training_time = model.train(X_train, y_train)
    
    # 4. Predict
    predictions, infer_time = model.predict(X_test)
    
    # 5. Descale
    preds = np.array(predictions)
    mean_scaled = preds[:, 0]
    std_scaled = preds[:, 3]
    lower95_scaled = preds[:, 1]
    upper95_scaled = preds[:, 2]
    ground_truth, mean, std, lower95, upper95 = descale_data(
        y_test, mean_scaled, std_scaled, lower95_scaled, upper95_scaled, scaler_mean, scaler_scale
    )
    
    # 6. Evaluate metrics
    rmse = ErrorMetrics.RMSE(mean, ground_truth)
    
    # 7. Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = dict(
        name="PCA-RGaSP",
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
    print(f"[PCA-RGaSP] metrics → {args.output_dir}/metrics.json")

if __name__=="__main__":
    main()