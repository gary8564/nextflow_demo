import numpy as np
import torch
import time
import gpytorch
from tqdm import tqdm

class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, latent_dim=2):
        super().__init__(
            torch.nn.Linear(data_dim, 1000),
            torch.nn.GELU(),
            torch.nn.Linear(1000, 500),
            torch.nn.GELU(),
            torch.nn.Linear(500, 50),
            torch.nn.GELU(),
            torch.nn.Linear(50, latent_dim),
        )

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type: str = "matern_5_2"):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        _, n_dimensions = train_x.shape
        mu_0 = 0.0
        sigma_0 = 1.0
        
        if kernel_type == 'matern_5_2':
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_dimensions,
                                                        lengthscale_prior=gpytorch.priors.LogNormalPrior(
                                                            mu_0 + np.log(n_dimensions) / 2, sigma_0
                                                            ),
                                                        )
        elif kernel_type == 'matern_3_2':
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=n_dimensions,
                                                        lengthscale_prior=gpytorch.priors.LogNormalPrior(
                                                            mu_0 + np.log(n_dimensions) / 2, sigma_0
                                                            ),
                                                        )
        elif kernel_type == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=n_dimensions, 
                                                    lengthscale_prior=gpytorch.priors.LogNormalPrior(
                                                        mu_0 + np.log(n_dimensions) / 2, sigma_0
                                                        ),
                                                    )
        else:
            raise ValueError("Unsupported kernel_type: {}".format(kernel_type))
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DKL_GPRegressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, reduced_dim: int, kernel_type: str = "matern_5_2"):
        super(DKL_GPRegressor, self).__init__(train_x, train_y, likelihood)
        self.reduced_dim = reduced_dim
        if kernel_type == 'matern_5_2':
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.reduced_dim)
        elif kernel_type == 'matern_3_2':
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.reduced_dim)
        elif kernel_type == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.reduced_dim) 
        else:
            raise ValueError("Unsupported kernel_type: {}".format(kernel_type))
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(base_kernel),
            num_dims=self.reduced_dim, grid_size=50 
        )
        self.feature_extractor = FeatureExtractor(data_dim=train_x.size(-1), latent_dim=self.reduced_dim)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DKL_MultiOutputGPRegressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, reduced_dim: int, kernel_type='matern_5_2'):
        """
        Args:
            train_x: training inputs of shape (n, d)
            train_y: training outputs of shape (n, num_tasks)
            likelihood: a MultitaskGaussianLikelihood instance
            kernel_type: one of 'matern_5_2', 'matern_3_2', or 'rbf'
            isotropic: whether to use a single lengthscale (True) or ARD (False)
        """
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = train_y.shape[-1]
        self.input_dim = train_x.shape[-1]
        self.reduced_dim = reduced_dim
        
        if kernel_type == 'matern_5_2':
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.reduced_dim, batch_shape=torch.Size([train_y.shape[1]]))
        elif kernel_type == 'matern_3_2':
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.reduced_dim, batch_shape=torch.Size([train_y.shape[1]]))
        elif kernel_type == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.reduced_dim, batch_shape=torch.Size([train_y.shape[1]])) 
        else:
            raise ValueError("Unsupported kernel_type: {}".format(kernel_type))

        # Create batched mean: ConstantMean with batch_shape = (num_tasks,)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_y.shape[1]]))
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=torch.Size([train_y.shape[1]]))
        self.feature_extractor = FeatureExtractor(data_dim=train_x.size(-1), latent_dim=self.reduced_dim)

    def forward(self, x):
        projected_x = self.feature_extractor(x) # x has shape (n, d)
        # The batched mean will have shape (num_tasks, n)
        mean_x = self.mean_module(projected_x)
        # The batched covariance will have shape (num_tasks, n, n)
        covar_x = self.covar_module(projected_x)
        # Construct a multitask distribution from the batch of independent GPs
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
        
class ExactGP:
    def __init__(self, device: str, kernel_type: str = 'matern_5_2'):
        self.device = torch.device(device)
        self.kernel_type = kernel_type
        
    def _train_loop(self, num_epochs: int, optim: str, lr: float, enable_scheduler: bool, loss_fn, model_params):
        # Optimizer
        fine_tune_optimizer = None
        if optim == "adam":
            optimizer = torch.optim.Adam(model_params, lr=lr)
        elif optim == "adamw":
            optimizer = torch.optim.AdamW(model_params, lr=lr)
        elif optim == "lbfgs":
            optimizer = torch.optim.LBFGS(model_params, lr=lr, max_iter=100)
        elif optim == "mixed":
            optimizer = torch.optim.Adam(model_params, lr=lr)
            fine_tune_optimizer = torch.optim.LBFGS(model_params, lr=0.1, max_iter=100)
        else:
            raise ValueError("Optimizer are only supported with `adam`, `adamw`, `lbfgs`, or `mixed`(i.e. Adam + LBFGS).")
        
        # Scheduler
        scheduler = None
        if enable_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
            
        def closure():
            optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -loss_fn(output, self.train_y)
            loss.backward()
            return loss
        def fine_tune_closure():
            fine_tune_optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -loss_fn(output, self.train_y)
            loss.backward()
            return loss
        # Training
        num_epochs = num_epochs
        start_time = time.time()
        for epoch in tqdm(range(num_epochs), desc="training..."):
            if optim == "lbfgs":     
                optimizer.step(closure)
                if epoch % 10 == 0:
                   current_loss = closure().item()
                   print(f"Epoch {epoch}/{num_epochs}, Loss: {current_loss:.3f}")
            else:    
                optimizer.zero_grad()
                output = self.model(self.train_X)
                loss = -loss_fn(output, self.train_y)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.3f}")
                loss.backward()
                optimizer.step()
                
            if scheduler is not None:
                scheduler.step(loss.item())
            
        if fine_tune_optimizer is not None:
            # LBFGS fine-tuning after training with ADAM
            if enable_scheduler:
                fine_tune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fine_tune_optimizer, mode='min', factor=0.5, patience=5)
            num_finetune_epochs = 20
            for epoch in tqdm(range(num_finetune_epochs), desc="LBFGS fine-tuning"):
                fine_tune_optimizer.step(fine_tune_closure)
                fine_tune_scheduler.step(loss.item())
                current_loss = fine_tune_closure().item()
                print(f"Fine-tuning Epoch {epoch+1}/{num_finetune_epochs}, Loss: {current_loss:.3f}")
                if fine_tune_optimizer.param_groups[0]['lr'] < 1e-6:
                    print("Learning rate too small, stopping fine-tuning")
                    break
        training_time = time.time() - start_time
        return training_time
        
    def train(self,
              train_X: np.ndarray | torch.Tensor, 
              train_y: np.ndarray | torch.Tensor, 
              num_epochs: int = 1000, 
              lr: float = 0.1,
              optim: str = "adam", 
              enable_scheduler: bool = False):
        if isinstance(train_X, np.ndarray):
            self.train_X = torch.from_numpy(train_X).to(torch.float32)
        else:
            self.train_X = train_X
        if isinstance(train_y, np.ndarray):
            self.train_y = torch.from_numpy(train_y).to(torch.float32)
        else:
            self.train_y = train_y
        self.train_X, self.train_y = self.train_X.to(self.device), self.train_y.to(self.device)
        
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(self.train_X, self.train_y, self.likelihood, self.kernel_type).to(self.device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Training
        training_time = self._train_loop(num_epochs, optim, lr, enable_scheduler, mll, self.model.parameters())
        print(f"Training GP takes {training_time:.3f} s")
        return training_time

    def predict(self, test_X: np.ndarray | torch.Tensor):
        if isinstance(test_X, np.ndarray):
            self.test_X = torch.from_numpy(test_X).to(torch.float32)
        else:
            self.test_X = test_X
        self.test_X = self.test_X.to(self.device)
        self.model.eval()
        self.likelihood.eval()
        start_time = time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(self.test_X))
        infer_time = time.time() - start_time    
        mean = predictions.mean.cpu().detach().numpy()
        std = predictions.stddev.cpu().detach().numpy()
        lower, upper = predictions.confidence_region()
        lower = lower.cpu().detach().numpy()
        upper = upper.cpu().detach().numpy()
        return mean, std, lower, upper, infer_time
        
class DKL_GP(ExactGP):
    def __init__(self, reduced_dim: int, device: str, kernel_type: str = 'matern_5_2'):
        super().__init__(device, kernel_type)
        self.reduced_dim = reduced_dim
    
    def train(self, 
              train_X: np.ndarray | torch.Tensor, 
              train_y: np.ndarray | torch.Tensor, 
              num_epochs: int = 200, 
              lr: float = 0.01, 
              optim: str = "adam", 
              enable_scheduler: bool = False):
        if isinstance(train_X, np.ndarray):
            self.train_X = torch.from_numpy(train_X).to(torch.float32)
        else:
            self.train_X = train_X
        if isinstance(train_y, np.ndarray):
            self.train_y = torch.from_numpy(train_y).to(torch.float32)
        else:
            self.train_y = train_y
        self.train_X, self.train_y = self.train_X.to(self.device), self.train_y.to(self.device)
        
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = DKL_GPRegressor(self.train_X, self.train_y, self.likelihood, self.reduced_dim, self.kernel_type).to(self.device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Optimizer
        params_group = [
                {'params': self.model.feature_extractor.parameters()},
                {'params': self.model.covar_module.parameters()},
                {'params': self.model.mean_module.parameters()},
                {'params': self.model.likelihood.parameters()},
            ]
        training_time = self._train_loop(num_epochs, optim, lr, enable_scheduler, mll, params_group)
        print(f"Training DKL-GP takes {training_time:.3f} s")
        return training_time  
    
class DKL_MoGP:
    def __init__(self):
        pass
