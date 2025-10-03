import torch
import time
import numpy as np
import gpytorch
import logging
from sklearn.preprocessing import StandardScaler  
from tqdm import tqdm
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, MultitaskKernel
from gpytorch.kernels.keops import RBFKernel, MaternKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from high_dim_gp.optim import FullBatchLBFGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchIndependentMultioutputGPModel(ExactGP):
    """
    This model implements a Batch Independent Multioutput GP,
    where the outputs are treated as independent (in batch) but share
    the same hyperparameters. This mimics the PPGaSP approach (shared kernel,
    fixed identity task covariance) and greatly reduces computational cost.
    """
    def __init__(self, train_x, train_y, likelihood, kernel_type='matern_5_2'):
        """
        Args:
            train_x: training inputs of shape (n, d)
            train_y: training outputs of shape (n, num_tasks)
            likelihood: a MultitaskGaussianLikelihood instance
            kernel_type: one of 'matern_5_2', 'matern_3_2', or 'rbf'
            
        """
        super().__init__(train_x, train_y, likelihood)
        self.register_buffer('train_x_buf', train_x)  
        self.register_buffer('train_y_buf', train_y)
        
        self.num_tasks = train_y.shape[-1]
        self.input_dim = train_x.shape[-1]
        # Mean module for multitask model
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=self.num_tasks)
        
        if kernel_type == 'matern_5_2':
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=self.input_dim)
        elif kernel_type == 'matern_3_2':
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=self.input_dim)
        elif kernel_type == 'rbf':
            base_kernel = RBFKernel(ard_num_dims=self.input_dim)
        else:
            raise ValueError("Unsupported kernel_type: {}".format(kernel_type))

        # Wrap the base kernel with a ScaleKernel to include an output-scale parameter
        # shared_kernel = ScaleKernel(base_kernel, outputscale_prior=GammaPrior(2.0, 0.15))
        coregionalized_kernel = MultitaskKernel(base_kernel, num_tasks=self.num_tasks, rank=0)
        
        # Create the batched kernel (num_tasks,)
        # self.covar_module = ScaleKernel(
        #     shared_kernel, 
        #     batch_shape=torch.Size([self.num_tasks])
        # ) if self.mimic_ppgasp else ScaleKernel(
        #     base_kernel,
        #     batch_shape=torch.Size([self.num_tasks])
        # )
        self.covar_module = ScaleKernel(coregionalized_kernel)
        
        # Initialize lengthscales with dimension-aware values
        self._initialize_lengthscales(train_x, train_y)
        
        # Parameters are shared across tasks via MultitaskKernel(rank=0)
        # Tie the parameters across tasks so that each output uses the same hyperparameters.
        self.tie_parameters_across_tasks()
    
    def _compute_reference_prior(self):
        """
        Compute log reference prior on inverse lengthscales (beta = 1 / lengthscale).
        """
        a = 0.2
        n = self.train_x_buf.shape[0]
        p = self.input_dim
        b = (1.0 / (n ** (1.0 / p))) * (a + p)

        inner = self._get_inner_kernel(self.covar_module)
        lengthscales = inner.lengthscale.squeeze()
        beta = 1.0 / lengthscales.clamp_min(1e-12)

        log_prior_ls = (a + 1.0) * torch.log(beta).sum() - b * beta.sum()
        return log_prior_ls

    def _get_inner_kernel(self, kernel):
        """
        Peel off any ScaleKernel wrappers, then if it's a MultitaskKernel
        dive into its .data_covar_module, and repeat until we hit
        the actual RBF/Matern kernel.
        """
        while True:
            if isinstance(kernel, ScaleKernel):
                kernel = kernel.base_kernel
            elif isinstance(kernel, MultitaskKernel):
                kernel = kernel.data_covar_module
            else:
                break
        return kernel
        
    def _initialize_lengthscales(self, train_x, train_y):
        """Initialize lengthscales based on input data characteristics"""
        # Calculate feature ranges for better initialization
        feature_ranges = train_x.max(dim=0).values - train_x.min(dim=0).values
        # Find the actual MaternKernel or RBFKernel you wrapped above
        inner = self._get_inner_kernel(self.covar_module)
        n_samples = train_x.shape[0]
        input_dim = float(self.input_dim)
        # Initialize lengthscales using empirical scaling
        scaling_factors = feature_ranges / (n_samples ** (1.0 / input_dim))
        init_lengthscales = 3.0 * scaling_factors
        inner.initialize(lengthscale=init_lengthscales.unsqueeze(0))
        # Set lower bounds for lengthscales in raw space
        lower_bounds = -torch.log(torch.tensor(0.1)) / (feature_ranges * input_dim)
        raw_lb = torch.log(torch.exp(lower_bounds) - 1.0)
        inner.register_constraint("raw_lengthscale", GreaterThan(raw_lb))
        # Initialize outputscale
        init_outputscale = train_y.var(dim=0).mean().sqrt()
        self.covar_module.outputscale = init_outputscale

    def tie_parameters_across_tasks(self):
        """
        Force the kernel parameters to be shared across all tasks.
        This mimics the PPGaSP assumption that the task covariance is the identity.
        """
        # Tie the lengthscale
        inner_kernel = self._get_inner_kernel(self.covar_module)
        base_lengthscale = inner_kernel.lengthscale.detach().clone()
        # common_lengthscale = base_lengthscale.mean(dim=0)
        # inner_kernel.lengthscale.data = common_lengthscale.unsqueeze(0).expand(self.num_tasks, 1, common_lengthscale.size(0)).clone()
        common_lengthscale = base_lengthscale.mean(dim=0, keepdim=True)
        inner_kernel.lengthscale.data = common_lengthscale.expand_as(base_lengthscale)
        
        # Tie the outputscale
        # base_outputscale = self.covar_module.outputscale.detach().clone()
        # common_outputscale = base_outputscale.mean()
        # self.covar_module.outputscale.data = common_outputscale.expand_as(self.covar_module.outputscale).clone()

        # # Tie the mean constant
        # base_mean = self.mean_module.constant.detach().clone()
        # common_mean = base_mean.mean()
        # self.mean_module.constant.data = common_mean.expand_as(self.mean_module.constant).clone()
          
    def forward(self, x):
        # x has shape (n, d)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)
            
class BiGP:
    def __init__(self, 
                 device: str, 
                 kernel_type: str = 'matern_5_2', 
                 method: str = 'post_mode',
                 prior_weight: float = 0.25,
                 normalize: bool = True):
        self.device = torch.device(device)
        self.kernel_type = kernel_type
        self.method = method  # 'post_mode' or 'mle'
        self.prior_weight = float(prior_weight)
        self.normalize = bool(normalize)
    
    def train(self,
              train_X: np.ndarray | torch.Tensor, train_Y: np.ndarray | torch.Tensor, 
              num_epochs: int = 200, lr: float = 0.1, 
              optim: str = "adam", num_finetune_epochs: int = 5):
        # Training setup
        if isinstance(train_X, np.ndarray):
            self.train_X = torch.from_numpy(train_X).to(torch.float32)
        else:
            self.train_X = train_X
        if isinstance(train_Y, np.ndarray):
            self.train_Y = torch.from_numpy(train_Y).to(torch.float32)
        else:
            self.train_Y = train_Y
        self.train_X, self.train_Y = self.train_X.to(self.device), self.train_Y.to(self.device)

        # Normalize X to [0, 1] and Y to zero-mean/unit-std
        if self.normalize:
            x_min = self.train_X.min(dim=0).values
            x_max = self.train_X.max(dim=0).values
            x_range = (x_max - x_min).clamp_min(1e-12)
            self.x_min = x_min
            self.x_range = x_range
            self.train_X = (self.train_X - self.x_min) / self.x_range

            y_mean = self.train_Y.mean(dim=0)
            y_std = self.train_Y.std(dim=0).clamp_min(1e-12)
            self.y_mean = y_mean
            self.y_std = y_std
            self.train_Y = (self.train_Y - self.y_mean) / self.y_std
        # Define the multitask likelihood.
        self.likelihood = MultitaskGaussianLikelihood(
            num_tasks=self.train_Y.shape[1],
            rank=0,
            noise_constraint=GreaterThan(1e-6),
        ).to(self.device)
        # Instantiate the improved batched model.
        self.model = BatchIndependentMultioutputGPModel(self.train_X, self.train_Y, self.likelihood,
                                                kernel_type=self.kernel_type).to(self.device)
        self.model.train()
        self.likelihood.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Optimizer
        fine_tune_optimizer = None
        if optim == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif optim == "lbfgs":
            optimizer = FullBatchLBFGS(self.model.parameters(), 
                                       lr=lr,
                                       history_size=10,
                                       line_search="Wolfe")
        elif optim == "mixed":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            fine_tune_optimizer = FullBatchLBFGS(self.model.parameters(), 
                                                lr=lr,
                                                history_size=10,
                                                line_search="Wolfe")
        else:
            raise ValueError("Optimizer are only supported with `adam`, `adamw`, `lbfgs`, or `mixed`(i.e. Adam + LBFGS).")
        
        # Training
        num_epochs = num_epochs
        start_time = time.time()
        
        if optim == "lbfgs":
            preconditioner_size = 100
            with gpytorch.settings.max_preconditioner_size(preconditioner_size):
                def closure():
                    optimizer.zero_grad()
                    output = self.model(self.train_X)
                    if self.method == 'post_mode':
                        objective = mll(output, self.train_Y) + self.prior_weight * self.model._compute_reference_prior()
                        loss = -objective
                    else:
                        loss = -mll(output, self.train_Y)
                    return loss
                loss = closure()
                loss.backward()
                
                for epoch in tqdm(range(num_epochs), desc="training..."):
                    options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                    loss, _, _, _, _, _, _, fail = optimizer.step(options)
                    
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.3f}")
                    
                    if fail:
                        print("LBFGS convergence reached!")
                        break
        else:    
            for epoch in tqdm(range(num_epochs), desc="training..."):
                optimizer.zero_grad()
                output = self.model(self.train_X)
                if self.method == 'post_mode':
                    objective = mll(output, self.train_Y) + self.prior_weight * self.model._compute_reference_prior()
                    loss = -objective
                else:
                    loss = -mll(output, self.train_Y)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.3f}")
                loss.backward()
                optimizer.step()
                            
        if fine_tune_optimizer is not None:
            # LBFGS fine-tuning after training with ADAM
            preconditioner_size = 100
            with gpytorch.settings.max_preconditioner_size(preconditioner_size):
                def fine_tune_closure():
                    fine_tune_optimizer.zero_grad()
                    output = self.model(self.train_X)
                    if self.method == 'post_mode':
                        objective = mll(output, self.train_Y) + self.prior_weight * self.model._compute_reference_prior()
                        loss = -objective
                    else:
                        loss = -mll(output, self.train_Y)
                    return loss
                loss = fine_tune_closure()
                loss.backward()
                for epoch in tqdm(range(num_finetune_epochs), desc="LBFGS fine-tuning"):
                    options = {'closure': fine_tune_closure, 'current_loss': loss, 'max_ls': 10}
                    loss, _, _, _, _, _, _, fail = fine_tune_optimizer.step(options)
                    
                    print(f"Fine-tuning Epoch {epoch+1}/{num_finetune_epochs}, Loss: {loss.item():.3f}")
                    
                    if fail:
                        print("LBFGS convergence reached!")
                        break
                    
                    if fine_tune_optimizer.param_groups[0]['lr'] < 1e-6:
                        print("Learning rate too small, stopping fine-tuning")
                        break
        
        training_time = time.time() - start_time
        print(f"Training GPytorch takes {training_time:.3f} s")
        return training_time
    
    def predict(self, test_X: np.ndarray | torch.Tensor):
        if isinstance(test_X, np.ndarray):
            self.test_X = torch.from_numpy(test_X).to(torch.float32)
        else:
            self.test_X = test_X
        self.test_X = self.test_X.to(self.device)

        if self.normalize:
            self.test_X = (self.test_X - self.x_min) / self.x_range
        self.model.eval()
        self.likelihood.eval()
        start_time = time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(self.test_X))
        infer_time = time.time() - start_time    
        mean = predictions.mean
        std = predictions.stddev
        lower, upper = predictions.confidence_region()

        # Denormalize Y if enabled
        if self.normalize:
            mean = mean * self.y_std + self.y_mean
            std = std * self.y_std
            lower = lower * self.y_std + self.y_mean
            upper = upper * self.y_std + self.y_mean

        mean = mean.cpu().detach().numpy()
        std = std.cpu().detach().numpy()
        lower = lower.cpu().detach().numpy()
        upper = upper.cpu().detach().numpy()
        return mean, std, lower, upper, infer_time
    
class PCA_BiGP(BiGP):
    def __init__(self, 
                 output_dim_reducer,
                 device: str, 
                 kernel_type: str = 'matern_5_2', 
                 method: str = 'post_mode'):
        super().__init__(device, kernel_type, method)
        self.output_dim_reducer = output_dim_reducer
        self.scaler = StandardScaler()
    
    def preprocess_dim_reduction(self, 
                                 train_X: np.ndarray,
                                 train_Y: np.ndarray, 
                                 test_X: np.ndarray, 
                                 test_Y: np.ndarray):
        # Store original dimensions
        self.original_input_dim = train_X.shape[1]
        self.original_output_dim = train_Y.shape[1]
        
        # Data standardization
        training_dataset = np.hstack((train_X, train_Y))
        training_dataset_scaled = self.scaler.fit_transform(training_dataset)
        test_dataset = np.hstack((test_X, test_Y))
        test_dataset_scaled = self.scaler.transform(test_dataset)
        train_X_scaled = training_dataset_scaled[:, :self.original_input_dim]
        train_Y_scaled = training_dataset_scaled[:, self.original_input_dim:]
        test_X_scaled = test_dataset_scaled[:, :self.original_input_dim]
        test_Y_scaled = test_dataset_scaled[:, self.original_input_dim:]
        
        # Apply output PCA 
        train_Y_scaled_reduced = self.output_dim_reducer.fit_transform(train_Y_scaled)
        # Verify reduced dimensions
        assert train_Y_scaled_reduced.shape[1] == self.output_dim_reducer.reducer.n_components, \
            f"Output PCA reduced to {train_Y_scaled_reduced.shape[1]} components, expected {self.output_dim_reducer.reducer.n_components}"
        print(f"Reduced output dimension to {train_Y_scaled_reduced.shape[1]}.")
        test_Y_scaled_reduced = self.output_dim_reducer.transform(test_Y_scaled)
        return train_X_scaled, train_Y_scaled_reduced, test_X_scaled, test_Y_scaled_reduced
    
    def postprocess_invert_back(self, 
                                predictions_mean: np.ndarray, 
                                predictions_std: np.ndarray = None):
        num_samples = predictions_mean.shape[0]
        # Transform reduced dimension back to original dimension 
        reconstructed_bands = self.output_dim_reducer.inverse_transform(predictions_mean)
        print(f"Inverse transform back to original output dimension: {reconstructed_bands.shape[1]}.")
        # Unnormalize data
        mu = self.scaler.mean_[self.original_input_dim:]
        sigma = self.scaler.scale_[self.original_input_dim:]
        predictions_mean_original = reconstructed_bands * sigma + mu
        std_original = None
        lower_CI = None
        upper_CI = None
        if predictions_std is not None:
            if hasattr(self.output_dim_reducer.reducer.model, "components_"):  
                W = self.output_dim_reducer.reducer.model.components_
                # Propagate diagonal covariance from latent space to original space:
                var_standardized = (predictions_std ** 2) @ (W ** 2)
                std_original = np.sqrt(var_standardized) * self.scaler.scale_[self.original_input_dim:]
            else:  
                num_mc_samples = 100
                reduced_dim = self.output_dim_reducer.reducer.n_components
                # Generate samples in reduced space
                rng = np.random.default_rng()
                latent_samples = rng.normal(loc=predictions_mean[:, :, None], scale=predictions_std[:, :, None], size=(num_samples, reduced_dim, num_mc_samples))
                # Reconstruct original output space
                original_dim = self.output_dim_reducer.reducer.model.n_features_in_
                original_samples = np.zeros((num_samples, original_dim, num_mc_samples))
                latent_flat = latent_samples.transpose(0, 2, 1).reshape(num_samples * num_mc_samples, reduced_dim)
                original_flat = self.output_dim_reducer.inverse_transform(latent_flat)
                original_flat = original_flat * sigma + mu
                original_samples = original_flat.reshape(num_samples, num_mc_samples, original_dim).transpose(0, 2, 1)
                # Compute statistics across samples
                std_original = np.std(original_samples, axis=2)
            margin = 1.96 * std_original
            lower_CI = predictions_mean_original - margin
            upper_CI = predictions_mean_original + margin 
        return predictions_mean_original, std_original, lower_CI, upper_CI
        
        
        

            