import os
import torch
import time
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gpytorch_emulator.utils import reduced_dim, viz_flattened_prediction, ErrorMetrics
from gpytorch_emulator import ExactGP, DKL_GP, DKL_MoGP
from psimpy.emulator import ScalarGaSP, PCAScalarGaSP, PPGaSP, PCAPPGaSP, LinearPCA, InputDimReducer

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

# Argparser
def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("High-dimensional input problem", add_help=add_help)
    parser.add_argument(
        "--case-study", 
        default="tsunami_tokushima", 
        type=str, 
        choices=["tsunami_tokushima", "synthetic"], 
        help="case study dataset for high dimensional input problem"
    )
    parser.add_argument(
        "--model",
        default="gyptorch",
        type=str,
        choices=["rgasp", "gpytorch"],
        help="gassian process model for high dimensional input problem"
    )
    parser.add_argument(
        "--output-dir",
        default="../results/tsunami_tokushima",
        type=str,
        help="Output directory to save results",
    )
    parser.add_argument(
        "--dim-reduction",
        action="store_true",
        help="Whether or not to reduce the input dimension."
    )
    return parser

def synthetic_function(X):
    """
    A simple function that returns a non-linear combination of the input parameters.
    Useful to test sensitivity analysis, as it should show a monotonically increasing sensitivity trend.
    The input parameter X can have any number of components > 55, while Y will be
    a vector of scalars.

    Parameters
    ----------
    X : numpy.ndarray
        Input array of shape (n_samples, n_vars). Must have n_vars >= 55.

    Returns
    -------
    Y : numpy.ndarray
        Output array of shape (n_samples,).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array")
    if X.shape[1] < 55:
        raise ValueError("uq_many_inputs_model is defined for inputs with more than 55 components")

    X = X.T  
    
    # Create weight matrix L: each row i has weight i+1
    L = np.tile(np.arange(1, X.shape[0] + 1)[:, np.newaxis], (1, X.shape[1]))

    # Compute the non-linear combination term-wise
    term1 = 3
    term2 = L * X**3
    term3 = (1/3) * L * np.log(X**2 + X**4)
    term4 = -5 * L * X
    Y = term1 + term2 + term3 + term4

    # Add extra sensitivity terms
    Y = (np.mean(Y, axis=0) + 
        X[0, :] * X[1, :]**2 - 
        X[4, :] * X[2, :] + 
        X[1, :] * X[3, :] + 
        X[50, :] + 
        X[49, :] * X[53, :]**2)

    Y = Y.T
    return Y

def generate_synthetc_input(n_samples=1000, n_vars=100, seed=None):
    """
    Generate synthetic input dataset by sampling from uniform distributions for each variable.

    By default, all variables are Uniform(1,2) except the 20th variable which is Uniform(1,3).

    Parameters
    ----------
    n_samples : int
        Number of samples to draw
    n_vars : int
        Number of input variables
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X : numpy.ndarray
        Array of shape (n_samples, n_vars) of sampled inputs.
    """
    if seed is not None:
        np.random.seed(seed)
    # Define lower and upper bounds
    low = np.ones(n_vars)
    high = np.ones(n_vars) * 2
    # Adjust the 20th variable (index 19)
    high[19] = 3
    # Draw samples
    X = np.random.uniform(low=low, high=high, size=(n_samples, n_vars))
    return X

def data_setup(case_study: str):
    if case_study == "tsunami_tokushima":
        # Tsunami
        data_root_folder = os.path.join(CURR_PATH, "../data/high_dim_input_prob/tsunami_Tokushima/")
        init_water_level = np.loadtxt(os.path.join(data_root_folder, "Initial_water_level_distibutions/deform4all.txt"), delimiter=',')
        input_dataset = init_water_level.copy()
        time_patterns = ["0100", "0250", "0500", "1200", "2400"]
        for pattern in time_patterns:
            tsunami_water_level = np.loadtxt(data_root_folder + "tsunami_water_level_disributions/z40" + pattern + "all_1d.txt", delimiter=",")
            input_dataset = np.hstack((input_dataset, tsunami_water_level))
        output_dataset = np.loadtxt(data_root_folder + "Inundation_distibutions/zmax5depth.txt", delimiter=',')
        selected_loc = int(np.median(np.argmax(output_dataset, axis=1)))
        X = input_dataset
        y = output_dataset[:, selected_loc]
        y = y[:, np.newaxis]    
    else:
        # 100D function
        X = generate_synthetc_input(n_samples=500)
        y = synthetic_function(X)
        y = y[:, np.newaxis]    
    dataset = np.hstack((X, y))
    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform(dataset)
    X_scaled, y_scaled = scaled_dataset[:, :-1], np.expand_dims(scaled_dataset[:, -1], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.15, shuffle=True)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_train, y_train, X_test, y_test, scaler

def descale_data(test_y, mean_scaled, std_scaled, lower95_scaled, upper95_scaled, standardizer: StandardScaler):
    mu = standardizer.mean_[-1] 
    sigma = standardizer.scale_[-1] 
    groud_truth = test_y * sigma + mu
    mean = mean_scaled * sigma + mu
    std = std_scaled * sigma 
    lower95 = lower95_scaled * sigma + mu
    upper95 = upper95_scaled * sigma + mu
    return groud_truth, mean, std, lower95, upper95

def scalargasp(train_X, train_y, test_X, test_y, standardizer):
    emulator = ScalarGaSP(ndim=train_X.shape[1])
    start_time = time.time()
    emulator.train(design=train_X, response=train_y)
    training_time = time.time() - start_time
    print(f"Training ScalarGaSP takes {training_time: .3f} s")
    start_time = time.time()
    predictions = emulator.predict(test_X)
    infer_time = time.time() - start_time
    print(f"Inference ScalarGaSP takes {infer_time: .3f} s")
    mean_scaled = predictions[:, 0]
    std_scaled = predictions[:, 3] 
    lower95_scaled = predictions[:, 1]
    upper95_scaled = predictions[:, 2] 
    ground_truth, mean, std, lower95, upper95 = descale_data(test_y, mean_scaled, std_scaled, lower95_scaled, upper95_scaled, standardizer)
    rmse = ErrorMetrics.RMSE(mean, ground_truth)
    coverage_prob = ErrorMetrics.CoverageProbability(mean, lower95, upper95, ground_truth)
    print(f'ScalarGaSP RMSE: {rmse:.3f}')
    print(f'ScalarGaSP Coverage probability: {coverage_prob:.3f}')
    return {
        "ground_truth": ground_truth,
        "prediction_mean": mean,
        "prediction_lower95": lower95,
        "prediction_upper95": upper95,
        "prediction_std": std,
        "rmse": rmse,
        "coverage_prob": coverage_prob,
        "training_time": training_time,
        "infer_time": infer_time
    }

def pca_scalargasp(train_X, train_y, test_X, test_y, standardizer):
    # Define the dimensionality reduction and Gaussian Process (GP) model
    pca = LinearPCA()
    input_reducer  = InputDimReducer(pca)
    model = PCAScalarGaSP(
        ndim=int(reduced_dim(input_reducer, train_X)),
        input_dim_reducer=input_reducer,
    )
    # Train the GP model
    training_time = model.train(train_X, train_y)
    
    # Inference
    predictions, infer_time = model.predict(test_X)    
    mean_scaled = predictions[:, 0]
    std_scaled = predictions[:, 3] 
    lower95_scaled = predictions[:, 1]
    upper95_scaled = predictions[:, 2] 
    ground_truth, mean, std, lower95, upper95 = descale_data(test_y, mean_scaled, std_scaled, lower95_scaled, upper95_scaled, standardizer)
    rmse = ErrorMetrics.RMSE(mean, ground_truth)
    coverage_prob = ErrorMetrics.CoverageProbability(mean, lower95, upper95, ground_truth)
    print(f'ScalarGaSP RMSE: {rmse:.3f}')
    print(f'ScalarGaSP Coverage probability: {coverage_prob:.3f}')
    return {
        "ground_truth": ground_truth,
        "prediction_mean": mean,
        "prediction_lower95": lower95,
        "prediction_upper95": upper95,
        "prediction_std": std,
        "rmse": rmse,
        "coverage_prob": coverage_prob,
        "training_time": training_time,
        "infer_time": infer_time
    }
    
def gpytorch(train_X, train_y, test_X, test_y, standardizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = ExactGP(device=device, kernel_type='matern_5_2')    
    training_time = emulator.train(train_X, train_y)
    mean_scaled, std_scaled, lower95_scaled, upper95_scaled, infer_time = emulator.predict(test_X)
    ground_truth, mean, std, lower95, upper95 = descale_data(test_y, mean_scaled, std_scaled, lower95_scaled, upper95_scaled, standardizer)
    rmse = ErrorMetrics.RMSE(mean, ground_truth)
    coverage_prob = ErrorMetrics.CoverageProbability(mean, lower95, upper95, ground_truth)
    print(f'ScalarGaSP RMSE: {rmse:.3f}')
    print(f'ScalarGaSP Coverage probability: {coverage_prob:.3f}')
    return {
        "ground_truth": ground_truth,
        "prediction_mean": mean,
        "prediction_lower95": lower95,
        "prediction_upper95": upper95,
        "prediction_std": std,
        "rmse": rmse,
        "coverage_prob": coverage_prob,
        "training_time": training_time,
        "infer_time": infer_time
    }
    
def dkl_gpytorch(train_X, train_y, test_X, test_y, standardizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = DKL_GP(reduced_dim=2, device=device, kernel_type='matern_5_2')
    training_time = emulator.train(train_X, train_y)
    mean_scaled, std_scaled, lower95_scaled, upper95_scaled, infer_time = emulator.predict(test_X)
    ground_truth, mean, std, lower95, upper95 = descale_data(test_y, mean_scaled, std_scaled, lower95_scaled, upper95_scaled, standardizer)
    rmse = ErrorMetrics.RMSE(mean, ground_truth)
    coverage_prob = ErrorMetrics.CoverageProbability(mean, lower95, upper95, ground_truth)
    print(f'ScalarGaSP RMSE: {rmse:.3f}')
    print(f'ScalarGaSP Coverage probability: {coverage_prob:.3f}')
    return {
        "ground_truth": ground_truth,
        "prediction_mean": mean,
        "prediction_lower95": lower95,
        "prediction_upper95": upper95,
        "prediction_std": std,
        "rmse": rmse,
        "coverage_prob": coverage_prob,
        "training_time": training_time,
        "infer_time": infer_time
    }
    
def main(args):
    if args.case_study not in ["tsunami_tokushima","synthetic"]:
        raise ValueError("""
                         This demo only provide two case studies: `tsunami_tokushima` and `synthetic`. If using custom dataset, data preprocessing will depend on the data file types.
                         After load the input/output data, follow the same logic as in this example code to train the GP emulator.
                         """)
    if args.model not in ["rgasp", "gpytorch"]:
        raise ValueError("Only `rgasp` and `gpytorch` are supported here as gaussian process emulator.")
    
    output_dir = os.path.join(CURR_PATH, args.output_dir)
    
    # ==== Load dataset ====
    train_X, train_y, test_X, test_y, standardizer = data_setup(args.case_study)
    
    # ======= ScalarGaSP =======
    if args.model == "rgasp":
        if args.dim_reduction:
            prediction_results = pca_scalargasp(train_X, train_y, test_X, test_y, standardizer)
            model_name = "pca_scalargasp"
        else:
            prediction_results = scalargasp(train_X, train_y, test_X, test_y, standardizer)
            model_name = "scalargasp"
    # ======== GP ========
    if args.model == "gpytorch":
        if args.dim_reduction:
            prediction_results = dkl_gpytorch(train_X, train_y, test_X, test_y, standardizer)
            model_name = "dkl_gpytorch"
        else:
            prediction_results = gpytorch(train_X, train_y, test_X, test_y, standardizer)
            model_name = "gpytorch"
    
    # Plot results
    test_y = prediction_results["ground_truth"]
    mean = prediction_results["prediction_mean"]
    std = prediction_results["prediction_std"]
    viz_flattened_prediction(test_y, mean, std, output_dir, model_name)
    
if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)

