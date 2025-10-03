import numpy as np
import sys
from warnings import warn
from scipy.stats import norm
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, root_mean_squared_error
from typing import Optional

def standard_normal_deviation_for_quantile_percent(quantile_percent: float) -> float:
    # central interval: p = 0.5 + 0.5 * (quantile / 100)
    p = 0.5 + 0.5 * (quantile_percent / 100.0)
    return float(norm.ppf(p))


def sample_from_predictive(
    mu: np.ndarray, sigma: np.ndarray, n_samples: int = 128, rng: np.random.Generator = None
) -> np.ndarray:
    """
    Draw samples from independent normals per output dim.
    mu, sigma: shape (P,), returns samples shape (n_samples, P)
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(loc=mu, scale=sigma, size=(n_samples, mu.shape[-1]))

class ErrorMetrics:
    @staticmethod
    def MAPE(predictions: np.ndarray, observations: np.ndarray) -> float:
        """Mean Absolute Percentage Error (MAPE)

        Args:
            predictions (np.ndarray): predictions
            measurements (np.ndarray): measurements

        Raises:
            ValueError: if sizes do not match

        Returns:
            float: error [%]
        """
        if predictions.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        return 100.0 * mean_absolute_percentage_error(
            y_true=observations, y_pred=predictions
        )

    @staticmethod
    def RMSE(predictions: np.ndarray, observations: np.ndarray) -> float:
        """Normalized Root-Mean-Square Error (NRMSE)

        Args:
            predictions (np.ndarray): predictions
            observations (np.ndarray): observations

        Raises:
            ValueError: if sizes do not match
            UserWarning: if mean of observations is zero

        Returns:
            float: NRMSD [1]
        """
        if predictions.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        RMSE = root_mean_squared_error(y_true=observations, y_pred=predictions)

        return RMSE

    @staticmethod
    def NRMSE(predictions: np.ndarray, observations: np.ndarray) -> float:
        """Normalized Root-Mean-Square Error (NRMSE)

        Args:
            predictions (np.ndarray): predictions
            observations (np.ndarray): observations

        Raises:
            ValueError: if sizes do not match
            UserWarning: if mean of observations is zero

        Returns:
            float: NRMSD [1]
        """
        if predictions.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        RMSE = np.sqrt(mean_squared_error(y_true=observations, y_pred=predictions))

        mean = np.mean(observations)
        if abs(mean) < sys.float_info.epsilon:
            warn("Mean of observations is zero (or very close to it)")

        return RMSE / mean

    @staticmethod
    def R2(predictions: np.ndarray, observations: np.ndarray) -> float:
        """R^2 Coefficient of Determination

        Args:
            predictions (np.ndarray): predictions
            observations (np.ndarray): observations

        Raises:
            ValueError: if sizes do not match

        Returns:
            float: R^2 error [1]
        """
        if predictions.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        return r2_score(y_true=observations, y_pred=predictions)

    @staticmethod
    def CoverageProbability(predictions_mean: np.ndarray, predictions_lower95: np.ndarray, predictions_upper95: np.ndarray, observations: np.ndarray) -> float:
        """Coverage Probability Calculation

        Args:
            predictions_mean (np.ndarray): maen values of predictions 
            predictions_lower95 (np.ndarray): lower 95% confidence interval of predictions
            predicttions_upper95 (np.ndarray): upper 95% confidence interval of predictions
            observations (np.ndarray): observations

        Raises:
            ValueError: if sizes do not match

        Returns:
            float: coverage probability [1]
        """
        
        if predictions_mean.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        return np.mean((predictions_upper95.flatten() >= observations.flatten()) & (observations.flatten() >= predictions_lower95.flatten()))

    @staticmethod
    def QuantileCoverageError(
        predictions_lower: np.ndarray,
        predictions_upper: np.ndarray,
        observations: np.ndarray,
        nominal_coverage: float = 0.95,
    ) -> float:
        """Quantile (interval) coverage error for central prediction intervals.

        Computes the absolute difference between the empirical coverage of the
        central interval [lower, upper] and the desired nominal coverage.
        This mirrors the idea of GPyTorch's quantile coverage metrics, adapted
        to the case where only the 2-sided central interval is available.

        Args:
            predictions_lower (np.ndarray): lower bounds of the interval
            predictions_upper (np.ndarray): upper bounds of the interval
            observations (np.ndarray): observed targets
            nominal_coverage (float): desired nominal coverage (e.g., 0.95)

        Returns:
            float: |empirical_coverage - nominal_coverage|

        Reference:
            GPyTorch metrics (quantile coverage) (`https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/metrics/metrics.py`).
        """
        lower = predictions_lower.astype(float).flatten()
        upper = predictions_upper.astype(float).flatten()
        y = observations.astype(float).flatten()

        if not (lower.size == upper.size == y.size):
            raise ValueError("lower, upper, and observations must have the same size")

        empirical_coverage = float(np.mean((y >= lower) & (y <= upper)))
        return abs(empirical_coverage - float(nominal_coverage))

    @staticmethod
    def MSLL(
        predictions_mean: np.ndarray,
        predictions_std: np.ndarray,
        observations: np.ndarray,
        train_y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Mean Standardized Log Loss (MSLL).

        Mirrors the definition used in GPyTorch where the mean log loss is
        standardized by subtracting the loss under a trivial model that predicts
        with the training data mean and variance.

        If ``train_y`` is ``None``, returns the mean log loss.

        Args:
            predictions_mean (np.ndarray): Predictive means.
            predictions_std (np.ndarray): Predictive standard deviations.
            observations (np.ndarray): Test targets.
            train_y (Optional[np.ndarray]): Training targets for standardization. 

        Returns:
            np.ndarray: MSLL per output task (if multi-output) or a scalar for single-output.

        References:
            - GPyTorch metrics implementation
              (`https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/metrics/metrics.py`).
        """
        # Determine the axis over which we average the pointwise log loss
        combine_dim = -2 if predictions_mean.ndim >= 2 else -1

        # Numerical stability for std, then square to variance
        predictions_variance = np.square(predictions_std)
        predictions_variance = np.clip(predictions_variance, a_min=1e-12, a_max=None)

        # Loss under model predictive distribution
        loss_model = 0.5 * np.log(2.0 * np.pi * predictions_variance) + (np.square(observations - predictions_mean) / (2.0 * predictions_variance))
        res = np.mean(loss_model, axis=combine_dim)

        if train_y is not None:
            data_mean = np.mean(train_y, axis=combine_dim)
            data_var = np.var(train_y, ddof=1)
            data_var = np.clip(data_var, a_min=1e-12, a_max=None)
            loss_trivial = 0.5 * np.log(2.0 * np.pi * data_var) + (
                np.square(observations - data_mean) / (2.0 * data_var)
            )
            res = res - np.mean(loss_trivial, axis=combine_dim)

        return res