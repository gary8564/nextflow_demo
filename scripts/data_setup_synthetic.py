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

def generate_synthetic_input(n_samples=1000, seed=None):
    """
    Generate synthetic input dataset by sampling from uniform distributions for each variable.

    By default, all variables are Uniform(1,2) except the 20th variable which is Uniform(1,3).

    Parameters
    ----------
    n_samples : int
        Number of samples to draw
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X : numpy.ndarray
        Array of shape (n_samples, n_vars) of sampled inputs.
    y : numpy.ndarray
        Array of shape (n_samples, 1) of sampled outputs.
    """
    if seed is not None:
        np.random.seed(seed)
    # Define lower and upper bounds
    low = np.ones(100)
    high = np.ones(100) * 2
    # Adjust the 20th variable (index 19)
    high[19] = 3
    # Draw samples
    X = np.random.uniform(low=low, high=high, size=(n_samples, 100))
    y = synthetic_function(X)
    y = y[:, np.newaxis]
    return X, y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True,
                   help="Store generated X.npy, y.npy")
    p.add_argument("--n-samples", type=int, default=500,
                   help="Number of samples to generate")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    args = p.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate synthetic data
    X, y = generate_synthetic_input(n_samples=args.n_samples, seed=args.seed)
    
    # Save data
    np.save(os.path.join(args.output_dir, "X.npy"), X)
    np.save(os.path.join(args.output_dir, "y.npy"), y)
    print(f"""[data_setup_synthetic] saved 
          input → {args.output_dir}/X.npy (shape: {X.shape})
          output → {args.output_dir}/y.npy (shape: {y.shape})""")

if __name__ == "__main__":
    main() 