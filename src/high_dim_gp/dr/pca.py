import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union

class BaseReducer(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """This method may raise an error if not supported by the particular algorithm."""
        raise NotImplementedError("Inverse transform not supported.")

class LinearPCA(BaseReducer):
    """ Linear Principal component analysis (PCA):
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.
    """
    def __init__(self, n_components: Union[int, str, float] = 0.95):
        """
        Args:
            n_components (Union[int, str, float], optional): Number of components to keep. Defaults to 0.95.
                    
                If n_components is `int` type and ``n_components > 0``, the number of components is simply as the specified n_components.
                
                If n_components is `float` type and ``0 < n_components < 1``, select the number of components such that the amount of variance that needs to be
                explained is greater than the percentage specified by n_components.
                
                If ``n_components == 'mle'``, the number of components is determined by the maximum likelihood estimation.
        """
        if n_components is None:
            raise TypeError("n_components must be int, float, or 'mle', not None")
        if not isinstance(n_components, (int, float, str)):
            raise TypeError("n_components must be int, float, or str")
        if isinstance(n_components, int) and n_components <= 0:
            raise AttributeError("If n_components is int type, it must be greater than zero.")
        if isinstance(n_components, str) and n_components != "mle":
            raise AttributeError("If n_components is str type, it must be specified as `mle`.")
        if isinstance(n_components, float) and (n_components <= 0 or n_components >= 1):
            raise AttributeError("If n_components is float type, 0 < n_components < 1.")
        self.n_components = n_components
        self.model = None

    def _compute_n_components(self, X: np.ndarray, explained_variance: float, show_cum_var_plot: bool) -> int:
        means = np.mean(X, axis=0)
        stds  = np.std(X, axis=0)
        tol = 0.01
        is_standardized = np.allclose(means, 0.0, atol=tol) and np.allclose(stds,  1.0, atol=tol)
        if not is_standardized:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X)
        # Cumulative variance
        num_components = np.arange(1, pca.n_components_ + 1, step=1)
        cum_varience = np.cumsum(pca.explained_variance_ratio_)
        # Get the number of PC whose varience > explained_variance
        k = np.argmax(cum_varience > explained_variance)
        print(f"Number of componenets with more than {explained_variance} of varience : {k + 1}.")
        
        # ensure step >= 1
        n = len(num_components)
        step = max(1, n // 10)
        if show_cum_var_plot:
            plt.figure()
            plt.title('Cumulative Explained Variance explained by the components')
            plt.ylabel('Cumulative Explained variance')
            plt.xlabel('Principal components')
            plt.xticks(np.arange(0, pca.n_components_ + 1, step=step))
            plt.axvline(x=k+1, color="k", linestyle="--")
            plt.axhline(y=0.95, color="r", linestyle="--")
            plt.plot(num_components, cum_varience)
            plt.ylim(0.0, 1.1)
            plt.show()
        return k + 1
        
    def _compute_explained_variance(self, X: np.ndarray, n_components: int, show_cum_var_plot: bool) -> float:
        means = np.mean(X, axis=0)
        stds  = np.std(X, axis=0)
        tol = 0.01
        is_standardized = np.allclose(means, 0.0, atol=tol) and np.allclose(stds,  1.0, atol=tol)
        if not is_standardized:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X)
        # Cumulative variance
        num_components = np.arange(1, pca.n_components_ + 1, step=1)
        cum_varience = np.cumsum(pca.explained_variance_ratio_)
        # Get the cumulative variance corresponds to n_components
        explained_variance = cum_varience[(n_components-1)]
        print(f"Explained variance when number of components is {n_components}: {explained_variance}.")
        
        # ensure step >= 1
        n = len(num_components)
        step = max(1, n // 10)
        if show_cum_var_plot:
            plt.figure()
            plt.title('Cumulative Explained Variance explained by the components')
            plt.ylabel('Cumulative Explained variance')
            plt.xlabel('Principal components')
            plt.xticks(np.arange(0, pca.n_components_ + 1, step=step))
            plt.axvline(x=n_components, color="r", linestyle="--")
            plt.axhline(y=explained_variance, color="k", linestyle="--")
            plt.plot(num_components, cum_varience)
            plt.ylim(0.0, 1.1)
            plt.show()
        return explained_variance

    def fit(self, X: np.ndarray, show_cum_var_plot: bool = False) -> None:
        """
        Fit the data to PCA model. 
        Noted that the maximum value of n_components can be is min(n_samples, n_features). 

        Args:
            X (np.ndarray): the 2d numpy array dataset with shape (n_samples, n_features).
        """
        if X.ndim != 2:
            raise ValueError(f"Input X must be 2d numpy array. But got shape {X.shape}.")
        n_samples, n_features = X.shape
        if isinstance(self.n_components, int) and self.n_components > min(n_samples, n_features):
            raise ValueError("n_components cannot exceed `min(n_samples, n_features)`.")   
        if isinstance(self.n_components, int):
            self.explained_variance = self._compute_explained_variance(X, self.n_components, show_cum_var_plot)
        elif isinstance(self.n_components, float):
            self.explained_variance = self.n_components
            self.n_components = self._compute_n_components(X, self.n_components, show_cum_var_plot)
        else:
            self.n_components = self.model.n_components_
            self.explained_variance = self._compute_explained_variance(X, self.n_components, show_cum_var_plot)
        self.model = PCA(n_components=self.n_components) 
        self.model.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA model is not fitted yet.")
        return self.model.transform(X)
    
    def fit_transform(self, X: np.ndarray, show_cum_var_plot: bool = False) -> np.ndarray:
        self.fit(X, show_cum_var_plot)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA model is not fitted yet.")
        return self.model.inverse_transform(X_reduced)

class NonlinearPCA(BaseReducer):
    def __init__(self,
                 n_components: int,
                 kernel: str = 'rbf',
                 fit_inverse_transform: bool = True,
                 alpha=0.001,
                 **kwargs):
        """
        Additional kwargs are forwarded to KernelPCA, for example gamma, degree, etc.
        """
        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']:
            raise ValueError("Kernel type is not supported in sklearn library. " \
            "Please check for the supported kernel types in the documentation of kernel PCA in scikit-learn.")
        self.n_components = n_components
        self.kernel = kernel
        self.fit_inverse_transform = fit_inverse_transform
        self.alpha = alpha
        self.kwargs = kwargs
        self.model: Optional[KernelPCA] = None

    def fit(self, X: np.ndarray) -> None:
        self.model = KernelPCA(n_components=self.n_components,
                               kernel=self.kernel,
                               fit_inverse_transform=self.fit_inverse_transform,
                               alpha=self.alpha,
                               **self.kwargs)
        self.model.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Kernel PCA model is not fitted yet.")
        return self.model.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Kernel PCA model is not fitted yet.")
        if not self.fit_inverse_transform:
            raise NotImplementedError("Kernel PCA was not configured for inverse_transform.")
        return self.model.inverse_transform(X_reduced)

class InputDimReducer:
    """Dimensionality reduction on input data."""
    def __init__(self, reducer: BaseReducer):
        self.reducer = reducer

    def fit(self, X: np.ndarray, show_cum_var_plot: bool = False) -> None:
        if X.size == 0:
            raise ValueError("Input data cannot be empty")
        if isinstance(self.reducer, LinearPCA):
            self.reducer.fit(X, show_cum_var_plot)
        else:
            self.reducer.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.transform(X)
    
    def fit_transform(self, X: np.ndarray, show_cum_var_plot: bool = False) -> np.ndarray:
        if X.size == 0:
            raise ValueError("Input data cannot be empty")
        if isinstance(self.reducer, LinearPCA):
            return self.reducer.fit_transform(X, show_cum_var_plot)
        else:
            return self.reducer.fit_transform(X)

class OutputDimReducer:
    """Dimensionality reduction on output data."""
    def __init__(self, reducer: BaseReducer):
        self.reducer = reducer

    def fit(self, Y: np.ndarray, show_cum_var_plot: bool = False) -> None:
        if Y.size == 0:
            raise ValueError("Output data cannot be empty")
        if isinstance(self.reducer, LinearPCA):
            self.reducer.fit(Y, show_cum_var_plot)
        else:
            self.reducer.fit(Y)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return self.reducer.transform(Y)
    
    def fit_transform(self, Y: np.ndarray, show_cum_var_plot: bool = False) -> np.ndarray:
        if Y.size == 0:
            raise ValueError("Output data cannot be empty")
        if isinstance(self.reducer, LinearPCA):
            return self.reducer.fit_transform(Y, show_cum_var_plot)
        else:
            return self.reducer.fit_transform(Y)

    def inverse_transform(self, Y_reduced: np.ndarray) -> np.ndarray:
        Y = self.reducer.inverse_transform(Y_reduced)
        return Y