import numpy as np
from beartype import beartype
from high_dim_gp.dr import InputDimReducer
import copy

@beartype
def reduced_dim(reducer: InputDimReducer | None, X: np.ndarray) -> int:
    """Helper function to obtain the input dimension

    Args:
        reducer (InputDimReducer | None): Dimensionality reduction
        X (np.ndarray): Input dataset

    Returns:
        int: number of input dimensions
    """
    if reducer is None:
        return X.shape[1]
    r = copy.deepcopy(reducer)
    r.reducer.fit(X, show_cum_var_plot=False)
    n_comp = r.reducer.n_components
    return int(n_comp) 
    