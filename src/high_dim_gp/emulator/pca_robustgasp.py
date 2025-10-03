import numpy as np
import time
from typing import Optional
from sklearn.preprocessing import StandardScaler
from psimpy.emulator import ScalarGaSP, PPGaSP
from high_dim_gp.dr import InputDimReducer, OutputDimReducer

class PCAScalarGaSP:
    def __init__(self,
                 ndim: int,
                 input_dim_reducer: Optional[InputDimReducer] = None,
                 **kwargs) -> None:
        """
        Set up basic parameters of the emulator with additional PCA for input dimensional reduction.

        Parameters
        ----------
        ndim : int
            Desired input parameter dimension after reduction.
        input_dim_reducer: Optional[InputDimReducer]
            An instance of InputDimReducer that will reduce the dimension of the input design matrix.
        **kwargs: 
            Additional parameters setting defined in RobustGaSPBase class.
        """
        self.ndim = ndim
        self.emulator = ScalarGaSP(ndim=self.ndim, **kwargs)
        self.input_dim_reducer = input_dim_reducer
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    def train(self, design: np.ndarray, response: np.ndarray, trend: Optional[np.ndarray] = None) -> None:
        if response.ndim == 1:
            response = np.reshape(response, (len(response), 1))
        elif response.ndim == 2 and response.shape[1] > 1:
            raise ValueError("PCAScalarGaSP only works for scalar-output model")
        design = self.input_scaler.fit_transform(design)
        response = self.output_scaler.fit_transform(response)
        # Ensure response is 1D for the R function
        response = response.ravel()
        if self.input_dim_reducer is not None:
            train_X = self.input_dim_reducer.fit_transform(design)
        else:
            train_X = design.copy()
        start_time = time.time()
        self.emulator.train(train_X, response, trend)
        training_time = (time.time() - start_time)
        print(f"Training PCAScalarGaSP takes {training_time:.3f} s")
        return training_time

    def predict(self, testing_input: np.ndarray, testing_trend: Optional[np.ndarray] = None) -> np.ndarray:
        testing_input = self.input_scaler.transform(testing_input)
        if self.input_dim_reducer is not None:
            test_X = self.input_dim_reducer.transform(testing_input)
        else:
            test_X = testing_input.copy()
        start_time = time.time()
        predictions_reduced = self.emulator.predict(test_X, testing_trend)
        infer_time = (time.time() - start_time)
        print(f"Inference PCAScalarGaSP takes {infer_time:.3f} s")
        predictions = self.output_scaler.inverse_transform(predictions_reduced)
        return predictions, infer_time

    def sample(self, testing_input: np.ndarray, nsamples: int = 1, testing_trend: Optional[np.ndarray] = None) -> np.ndarray:
        testing_input = self.input_scaler.transform(testing_input)
        if self.input_dim_reducer is not None:
            testing_input = self.input_dim_reducer.transform(testing_input)
        samples = self.emulator.sample(testing_input, nsamples, testing_trend)
        return samples
    
        
class PCAPPGaSP:
    def __init__(self,
                 ndim: int,
                 input_dim_reducer: Optional[InputDimReducer] = None,
                 output_dim_reducer: Optional[OutputDimReducer] = None,
                 **kwargs) -> None:
        """
        Set up basic parameters of the emulator with additional PCA for input and/or output dimensional reduction.

        Parameters
        ----------
        ndim : int
            Desired input parameter dimension after reduction.
        input_dim_reducer: Optional[InputDimReducer]
            An instance of InputDimReducer that will reduce the dimension of the input design matrix.
        output_reducer :
            An instance of OutputDimReducer that will reduce the dimension of the output response and
            later recover it via inverse_transform.
        **kwargs: 
            Additional parameters setting defined in RobustGaSPBase class.
        """
        zero_mean = kwargs.get("zero_mean", "No")
        if zero_mean == "Yes" and output_dim_reducer is not None:
            raise ValueError("Cannot use zero_mean='Yes' when PCA is used for output. PCA already centers the output.")
        self.ndim = ndim
        self.emulator = PPGaSP(ndim=self.ndim, **kwargs)
        self.input_dim_reducer = input_dim_reducer
        self.output_dim_reducer = output_dim_reducer
        self.original_input_dim = None
        self.original_output_dim = None
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
    def _postprocess_testing_output(self, predictions: np.ndarray, uncertainty_reconstruction: bool = False):
        """
        Post-process the PCA-GaSP predictions by converting the emulator's output
        in the reduced space back to the original output space.
        
        Parameters:
            predictions: 
                        Predicted results (mean, lower 95, upper 95, sd) of the trained emulator.
            uncertainty_reconstruction: 
                        whether or not to reconstruct the uncertainty quantification from latent reduced space to original dimension space.
    
        Returns:
            predictions_mean
            predictions_lower_95
            predictions_upper_95
            predictions_std
        """
        if not (predictions.ndim == 3):
            raise ValueError("predictions must be 3d numpy array.")
        
        num_test, reduced_dim, _ = predictions.shape

        if self.output_dim_reducer is None:
            return predictions[:, :, 0], predictions[:, :, 1], predictions[:, :, 2], predictions[:, :, 3]
                
        try:
            mean_reduced = predictions[:, :, 0]
            mean_original = self.output_dim_reducer.inverse_transform(mean_reduced)
            mean_original = self.output_scaler.inverse_transform(mean_original)
            lower_CI, upper_CI, std_original = None, None, None  
            if uncertainty_reconstruction:
                std_reduced = predictions[:, :, 3]
                if hasattr(self.output_dim_reducer.reducer.model, "components_"):  
                    W = self.output_dim_reducer.reducer.model.components_
                    # Propagate diagonal covariance from latent space to original space:
                    var_standardized = (std_reduced ** 2) @ (W ** 2)
                    std_original = np.sqrt(var_standardized) * self.output_scaler.scale_
                else:  
                    num_mc_samples = 100
                    # Generate samples in reduced space
                    rng = np.random.default_rng()
                    latent_samples = rng.normal(loc=mean_reduced[:, :, None], scale=std_reduced[:, :, None], size=(num_test, reduced_dim, num_mc_samples))
                    # Reconstruct original output space
                    original_dim = self.output_dim_reducer.reducer.model.n_features_in_
                    original_samples = np.zeros((num_test, original_dim, num_mc_samples))
                    latent_flat = latent_samples.transpose(0, 2, 1).reshape(num_test * num_mc_samples, reduced_dim)
                    original_flat = self.output_dim_reducer.inverse_transform(latent_flat)
                    original_flat = self.output_scaler.inverse_transform(original_flat)
                    original_samples = original_flat.reshape(num_test, num_mc_samples, original_dim).transpose(0, 2, 1)
                    # Compute statistics across samples
                    std_original = np.std(original_samples, axis=2)
                margin = 1.96 * std_original
                lower_CI = mean_original - margin
                upper_CI = mean_original + margin                         
            return mean_original, lower_CI, upper_CI, std_original
        except NotImplementedError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error in post-processing predictions: {str(e)}")    

    def train(self, design: np.ndarray, response: np.ndarray, trend: Optional[np.ndarray] = None) -> float:
        # Store original dimensions
        self.original_input_dim = design.shape[1]
        self.original_output_dim = response.shape[1]
        
        # Data standardization
        design = self.input_scaler.fit_transform(design)
        response = self.output_scaler.fit_transform(response)
        
        # Apply input PCA if specified
        if self.input_dim_reducer is not None:
            start_time = time.time()
            train_X = self.input_dim_reducer.fit_transform(design)
            input_pca_time = (time.time() - start_time)
            # Verify reduced dimensions
            assert train_X.shape[1] == self.input_dim_reducer.reducer.n_components, \
                f"Input PCA reduced to {train_X.shape[1]} components, but expected {self.input_dim_reducer.reducer.n_components}"
        else:
            input_pca_time = 0
            train_X = design.copy()
        # Verify initialized ndim parameters for emulator
        assert train_X.shape[1] == self.ndim, \
            f"Input dimension of train data is {train_X.shape[1]}, but expected dimension of the defined emulator is {self.ndim}"
        # Apply output PCA if specified
        if self.output_dim_reducer is not None:
            start_time = time.time()
            train_Y = self.output_dim_reducer.fit_transform(response)
            output_pca_time = (time.time() - start_time)
            # Verify reduced dimensions
            assert train_Y.shape[1] == self.output_dim_reducer.reducer.n_components, \
                f"Output PCA reduced to {train_Y.shape[1]} components, expected {self.output_dim_reducer.reducer.n_components}"
            # Cache latent training range for Monte Carlo trust-region
            self.latent_train_min = np.min(train_Y, axis=0)
            self.latent_train_max = np.max(train_Y, axis=0)
        else:
            output_pca_time = 0
            train_Y = response.copy()
        # Train the emulator
        start_time = time.time()
        self.emulator.train(train_X, train_Y, trend)
        gp_training_time = (time.time() - start_time)
        training_time = input_pca_time + output_pca_time + gp_training_time
        print(f"Training PCAPPGaSP takes {training_time:.3f} s")
        return training_time
        
    def predict(self, 
                testing_input: np.ndarray,
                testing_trend: Optional[np.ndarray] = None, 
                uncertainty_reconstruction: bool = False):
        """
        Args:
            testing_input (np.ndarray): input data for inference
            testing_trend (Optional[np.ndarray], optional): trend function. Defaults to None.
            uncertainty_reconstruction (bool, optional): whether to propagate the uncertainty quantification from latent reduced space to original dimension space. Defaults to False.
            
        Returns:
            predictions_latent: predictions in latent space 
                Shape :code:`(n_test, n_latent_dim, 4)`.
                `predictions[:, :, 0]` - mean,
                `predictions[:, :, 1]` - low95,
                `predictions[:, :, 2]` - upper95,
                `predictions[:, :, 3]` - sd
            predictions_mean_orig: predictions mean in original space
                Shape :code:`(n_test, n_original_dim)`.
            infer_time: inference time
        """
        # Validate input dimensions
        if testing_input.shape[1] != self.original_input_dim:
            raise ValueError(f"Expected input dimension {self.original_input_dim}, got {testing_input.shape[1]}")
        
        # Data standardization
        testing_input = self.input_scaler.transform(testing_input)
            
        # Reduce input dimension if needed
        if self.input_dim_reducer is not None:
            test_X = self.input_dim_reducer.transform(testing_input)
            assert test_X.shape[1] == self.input_dim_reducer.reducer.n_components, \
                f"Transformed input has {test_X.shape[1]} components, expected {self.input_dim_reducer.reducer.n_components}"
        else:
            test_X = testing_input.copy()
        # Verify initialized ndim parameters for emulator
        assert test_X.shape[1] == self.ndim, \
            f"Input dimension of test data is {test_X.shape[1]}, but expected dimension of the trained emulator is {self.ndim}"
        
        try:
            # Get predictions
            start_time = time.time()
            predictions_latent = self.emulator.predict(testing_input=test_X, testing_trend=testing_trend)

            # Post-process predictions
            predictions_mean_orig, lower_CI, upper_CI, std_orig = self._postprocess_testing_output(predictions_latent, uncertainty_reconstruction)            
            infer_time = (time.time() - start_time)
            print(f"Inference PCAPPGaSP takes {infer_time:.3f} s")
            
            if lower_CI is not None and upper_CI is not None and std_orig is not None:
                predictive_uncertainties = np.dstack([lower_CI, upper_CI, std_orig])
            else:
                predictive_uncertainties = None
            
            # Verify output dimensions
            if self.output_dim_reducer is not None:
                assert predictions_mean_orig.shape[1] == self.original_output_dim, \
                    f"Output has {predictions_mean_orig.shape[1]} dimensions, expected {self.original_output_dim}"
                
            return predictions_latent, predictions_mean_orig, infer_time, predictive_uncertainties
        
        except NotImplementedError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error in prediction: {str(e)}")

    def sample(self, testing_input: np.ndarray, nsamples: int = 1, testing_trend: Optional[np.ndarray] = None) -> np.ndarray:
        testing_input = self.input_scaler.transform(testing_input)
        if self.input_dim_reducer is not None:
            testing_input = self.input_dim_reducer.transform(testing_input)
        samples = self.emulator.sample(testing_input, nsamples, testing_trend)
        if self.output_dim_reducer is not None:
            ntest, n_reduced, _ = samples.shape
            original_dim = self.output_dim_reducer.reducer.model.n_features_in_
            samples_flat = samples.transpose(0, 2, 1).reshape(ntest * nsamples, n_reduced)
            orig_flat = self.output_dim_reducer.inverse_transform(samples_flat)
            orig_flat = self.output_scaler.inverse_transform(orig_flat)
            samples_original = orig_flat.reshape(ntest, nsamples, original_dim).transpose(0, 2, 1)
            return samples_original
        else:
            return samples
    
    def get_range_param(self):
        return self.emulator.get_range_param()
    
    def get_trend_param(self):
        return self.emulator.get_trend_param()

