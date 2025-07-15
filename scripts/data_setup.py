import argparse
import os
import numpy as np

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


def load_tsunami(data_root_folder):
    """Load tsunami files from the given root folder."""
    init_water_level = np.loadtxt(os.path.join(data_root_folder,
        "Initial_water_level_distibutions/deform4all.txt"), delimiter=",")
    X = init_water_level.copy()
    time_patterns = ["0100", "0250", "0500", "1200", "2400"]
    for pattern in time_patterns:
        tsunami_water_level = np.loadtxt(
                                os.path.join(data_root_folder,
                                    f"tsunami_water_level_disributions/z40{pattern}all_1d.txt"
                                ), delimiter=","
                              )
        X = np.hstack((X, tsunami_water_level))
    Y_all = np.loadtxt(os.path.join(data_root_folder,
        "Inundation_distibutions/zmax5depth.txt"), delimiter=",")
    # pick a representative station (median of per-sample argmax)
    loc = int(np.median(np.argmax(Y_all, axis=1)))
    y = Y_all[:, loc][:, None]
    return X, y

def data_setup(case_study: str, input_dir: str = None):
    if case_study == "tsunami_tokushima":
        if input_dir is None:
            raise ValueError("input_dir cannot be None for case study tsunami_tokushima.")
        X, y = load_tsunami(input_dir)   
    else:
        # 100D function
        X = generate_synthetc_input(n_samples=500)
        y = synthetic_function(X)
        y = y[:, np.newaxis]
    return X, y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case-study",
                   choices=["synthetic", "tsunami_tokushima"],
                   required=True)
    p.add_argument("--input-dir",
                   help="(tsunami only) directory where gdown+unzip placed the files")
    p.add_argument("--output-dir", required=True,
                   help="Store intermediate X.npy, y.npy")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    X, y = data_setup(args.case_study)
    np.save(os.path.join(args.output_dir,"X.npy"), X)
    np.save(os.path.join(args.output_dir,"y.npy"), y)
    print(f"[data_setup] saved → {args.output_dir}/{{X.npy,y.npy}}")

if __name__=="__main__":
    main()

