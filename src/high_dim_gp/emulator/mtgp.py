import torch
import gpytorch
import numpy as np
import logging
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVGP_LMC(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, num_tasks, rank=50, num_inducing=128, use_pca_init=True):
        self.rank = rank
        self.num_tasks = num_tasks
        self.use_pca_init = use_pca_init
        
        # Set up inducing points for the latent GPs
        inducing_points = torch.randn(rank, num_inducing, input_dim)
        
        # Use LMCVariationalStrategy
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            base_variational_strategy=gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                gpytorch.variational.CholeskyVariationalDistribution(
                    num_inducing, batch_shape=torch.Size([rank])
                ),
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=rank,
            latent_dim=-1
        )
        
        super().__init__(variational_strategy)

        # Mean & kernel for the latent GPs
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([rank]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=input_dim,
                lengthscale_constraint=gpytorch.constraints.Positive()
            ),
            outputscale_constraint=gpytorch.constraints.Positive()
        )

    def _init_mixing_matrix_from_pca(self, train_Y):
        """Initialize the LMC mixing matrix using PCA"""
        if not self.use_pca_init:
            return
            
        logger.info("Initializing LMC mixing matrix with PCA...")
        # PCA components cannot exceed min(n_samples, n_features)
        n_samples, n_features = train_Y.shape
        max_components = min(n_samples - 1, n_features, self.rank)  # n_samples - 1 for numerical stability
        
        if max_components < self.rank:
            print(f"Warning: Reducing PCA components from {self.rank} to {max_components} due to sample size")
        
        # Perform PCA on the output data
        pca = PCA(n_components=max_components)
        pca.fit(train_Y.detach().numpy())
        
        # Use PCA components to initialize the LMC coefficients
        with torch.no_grad():
            # PCA components are [max_components, n_features], we need [num_latents, num_tasks]
            pca_components = torch.tensor(pca.components_, dtype=torch.float32)  # [max_components, num_tasks]
            
            # If we have fewer PCA components than rank, pad with random values
            if max_components < self.rank:
                additional_components = torch.randn(self.rank - max_components, n_features) * 0.1
                pca_components = torch.cat([pca_components, additional_components], dim=0)
            
            # lmc_coefficients expects shape [num_latents, num_tasks]
            self.variational_strategy.lmc_coefficients.data = pca_components
            
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:10]}...")  # Show first 10
        logger.info(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        logger.info(f"LMC coefficients initialized with shape: {self.variational_strategy.lmc_coefficients.shape}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultiTaskGP:
    """
    Multi-task Gaussian Process using Sparse Variational GP with Linear Model of Coregionalization (LMC).
    
    This class automatically determines optimal model parameters (rank and num_inducing) based on 
    the training data characteristics, using base parameters as guidance.
    
    Parameters:
    -----------
    base_inducing : int, default=64
        Base number of inducing points. The actual number will be calculated based on 
        data size, dimensionality, and complexity. Larger values lead to more inducing 
        points but higher computational cost.
        
    min_rank : int, default=10
        Minimum rank for the Linear Model of Coregionalization. The actual rank will 
        be determined based on PCA analysis of the output data, but will not go below 
        this minimum value.
        
    use_pca_init : bool, default=True
        Whether to initialize the LMC mixing matrix using PCA components from the 
        output data. This typically leads to better initialization and faster convergence.
        
    device : str, default="cpu"
        Device to run computations on ("cpu" or "cuda").
        
    Attributes:
    -----------
    rank : int
        Actual rank used for LMC (set after calling train())
        
    num_inducing : int  
        Actual number of inducing points used (set after calling train())
        
    Notes:
    ------
    The rank is automatically determined based on PCA analysis to capture ~90% of 
    output variance, with constraints based on output dimensionality and the min_rank parameter.
    
    The number of inducing points is calculated based on:
    - base_inducing as foundation
    - Training data size 
    - Input dimensionality
    - Computational constraints
    """
    def __init__(self, base_inducing=64, min_rank=10, use_pca_init=True, device="cpu"):
        self.base_inducing = base_inducing  # Base number for inducing point calculation
        self.min_rank = min_rank  # Minimum rank for LMC
        self.use_pca_init = use_pca_init
        self.device = torch.device(device)
        self.rank = None
        self.num_inducing = None
        
    # Beta scheduler for geophysical data
    def _get_beta(self, epoch):
        # Very gradual annealing - critical for geophysical applications
        warmup = 50  # Extended warmup
        ramp_epochs = 200  # Slower ramp
        if epoch < warmup:
            return 0.001  # Very small initial beta
        elif epoch < warmup + ramp_epochs:
            progress = (epoch - warmup) / ramp_epochs
            # Smooth sigmoid transition
            return 0.001 + 0.999 * (1 / (1 + np.exp(-10 * (progress - 0.5))))
        else:
            return 1.0
    
    # Learning rate scheduling
    def _get_lr(self, epoch, total_epochs):
        warmup_epochs = 20
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            # Cosine annealing after warmup
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    def _normalize_data(self, train_X, train_Y):
        """Normalize input and output data"""
        logger.info("Normalizing data...")
        self.train_X_mean = train_X.mean(dim=0, keepdim=True)
        self.train_X_std = train_X.std(dim=0, keepdim=True) + 1e-6
        self.train_X = (train_X - self.train_X_mean) / self.train_X_std
        
        self.train_Y_mean = train_Y.mean(dim=0, keepdim=True)
        self.train_Y_std = train_Y.std(dim=0, keepdim=True) + 1e-6
        self.train_Y = (train_Y - self.train_Y_mean) / self.train_Y_std
        
        return self.train_X, self.train_Y

    def _determine_model_parameters(self, train_Y, input_dim, output_dim):
        """Determine optimal rank and inducing points based on data characteristics"""
        n_samples = self.train_X.shape[0]
        logger.info(f"Problem size: {input_dim} inputs → {output_dim} outputs")
        logger.info(f"Training samples: {n_samples}")
        logger.info(f"Base parameters: base_inducing={self.base_inducing}, min_rank={self.min_rank}")
        
        # Adaptive rank selection based on data characteristics
        pca = PCA()
        pca.fit(train_Y.detach().numpy())
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # More conservative rank selection for geophysical data
        r_90 = np.searchsorted(cumulative_variance, 0.90) + 1  
        self.rank = min(max(r_90, self.min_rank), output_dim // 2)  
        
        # Adaptive inducing point count based on data size and complexity
        # Use base_inducing as the foundation for calculation
        base_calculation = min(256, max(self.base_inducing, n_samples // 4))
        # Scale with input dimensionality for geophysical complexity
        self.num_inducing = min(512, base_calculation + input_dim * 8)
        
        # Ensure we don't have more inducing points than we can reasonably support
        max_reasonable_inducing = min(512, n_samples + 50)  # Allow some buffer
        if self.num_inducing > max_reasonable_inducing:
            self.num_inducing = max_reasonable_inducing
            
        logger.info(f"Calculated parameters: rank={self.rank}, num_inducing={self.num_inducing}")
        logger.info(f"Explained variance at rank {self.rank}: {cumulative_variance[self.rank-1]:.3f}")

    def _validate_parameters(self):
        """Ensure that rank and num_inducing have been properly set"""
        if self.rank is None or self.num_inducing is None:
            raise RuntimeError("Model parameters (rank, num_inducing) must be determined before initialization. "
                             "Call _determine_model_parameters first.")

    def _initialize_inducing_points(self):
        """Initialize inducing points using k-means clustering"""
        self._validate_parameters()
        logger.info("Initializing inducing points with k-means clustering...")
        
        n_samples, input_dim = self.train_X.shape
        
        if n_samples >= self.num_inducing:
            # Use k-means clustering
            kmeans = KMeans(n_clusters=self.num_inducing, random_state=42, n_init=10)
            cluster_centers = kmeans.fit(self.train_X.detach().cpu().numpy()).cluster_centers_
            # Shape: [num_inducing, input_dim]
            inducing_points_2d = torch.tensor(cluster_centers, dtype=torch.float32, device=self.device)
        else:
            logger.warning(f"Only {n_samples} samples available, but need {self.num_inducing} inducing points")
            available_points = self.train_X.clone()  # [n_samples, input_dim]
            
            # Pad with random points based on data distribution
            data_mean = self.train_X.mean(dim=0, keepdim=True)  # [1, input_dim]
            data_std = self.train_X.std(dim=0, keepdim=True) + 1e-6  # [1, input_dim]
            n_random = self.num_inducing - n_samples
            random_points = torch.randn(n_random, input_dim, device=self.device) * data_std + data_mean
            inducing_points_2d = torch.cat([available_points, random_points], dim=0)
        
        # Expand to shape [rank, num_inducing, input_dim]
        inducing_points = inducing_points_2d.unsqueeze(0).repeat(self.rank, 1, 1)
        
        logger.info(f"Initialized inducing points with shape: {inducing_points.shape}")
        return inducing_points

    def _setup_model(self, input_dim, output_dim, inducing_points):
        """Setup the SVGP_LMC model and set inducing points"""
        self._validate_parameters()
        
        # Validation - inducing points should already have the correct shape
        expected_shape = (self.rank, self.num_inducing, input_dim)
        if inducing_points.shape != expected_shape:
            raise RuntimeError(
                f"Inducing points shape mismatch: got {inducing_points.shape}, "
                f"expected {expected_shape}. This indicates a bug in _initialize_inducing_points()."
            )
                    
        self.model = SVGP_LMC(input_dim, output_dim, rank=self.rank, num_inducing=self.num_inducing)
        self.model.to(self.device)
        # Set inducing points
        with torch.no_grad():
            self.model.variational_strategy.base_variational_strategy.inducing_points.data = inducing_points
        
        # Mixing matrix initialization
        self.model._init_mixing_matrix_from_pca(self.train_Y)

    def _initialize_kernel_parameters(self, input_dim):
        """Initialize kernel parameters including lengthscale and outputscale"""
        with torch.no_grad():
            # Initialize length scales based on input range and dimensionality
            input_range = self.train_X.max(dim=0)[0] - self.train_X.min(dim=0)[0]
            initial_lengthscale = input_range.mean() / np.sqrt(input_dim)  # Scale with dimensionality
            
            # Get the current lengthscale shape and initialize properly
            current_lengthscale = self.model.covar_module.base_kernel.lengthscale
            if current_lengthscale.dim() == 1:
                # Single lengthscale per latent GP
                self.model.covar_module.base_kernel.lengthscale = initial_lengthscale.expand_as(current_lengthscale)
            else:
                # ARD lengthscales: [rank, input_dim] or [rank, 1, input_dim]
                target_shape = current_lengthscale.shape
                if len(target_shape) == 2:  # [rank, input_dim]
                    lengthscale_init = initial_lengthscale.expand(target_shape[0], target_shape[1])
                elif len(target_shape) == 3:  # [rank, 1, input_dim]
                    lengthscale_init = initial_lengthscale.expand(target_shape[0], target_shape[1], target_shape[2])
                else:
                    # Fallback: just use the scalar value
                    lengthscale_init = initial_lengthscale.expand_as(current_lengthscale)
                self.model.covar_module.base_kernel.lengthscale = lengthscale_init

    def _initialize_outputscale(self):
        """Initialize output scale parameters"""
        with torch.no_grad():
            current_outputscale = self.model.covar_module.outputscale
            self.model.covar_module.outputscale = torch.ones_like(current_outputscale) * 0.5

    def _initialize_mean(self):
        """Initialize mean module parameters"""
        with torch.no_grad():
            output_range = self.train_Y.std(dim=0).mean()
            current_mean = self.model.mean_module.constant
            self.model.mean_module.constant = torch.zeros_like(current_mean) * output_range * 0.1

    def _initialize_likelihood(self, output_dim):
        """Initialize likelihood with proper noise settings"""
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=output_dim,
            noise_constraint=gpytorch.constraints.Interval(1e-5, 0.5),  
            has_task_noise=True,
            has_global_noise=True
        )
        self.likelihood.to(self.device)

    def _initialize_noise(self, output_dim):
        """Initialize noise parameters"""
        noise_init = 0.05 * torch.ones(output_dim)  # Smaller initial noise
        with torch.no_grad():
            if self.likelihood.has_task_noise and self.likelihood.rank == 0:
                self.likelihood.task_noises = noise_init
            if self.likelihood.has_global_noise:
                self.likelihood.noise = 0.01  # Small global noise

    def _setup_optimizer_and_scheduler(self, total_epochs, lr, enable_scheduler):
        """Setup optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=lr)
        if enable_scheduler:
            # Create lambda function that captures total_epochs
            lr_lambda = lambda epoch: self._get_lr(epoch, total_epochs)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = None
        return optimizer, scheduler

    def _training_loop(self, optimizer, scheduler, epochs):
        """Main training loop with early stopping"""
        logger.info("Starting training...")
        
        # Validate tensor shapes before training
        logger.info(f"Training data shapes: X={self.train_X.shape}, Y={self.train_Y.shape}")
        logger.info(f"Model parameters: rank={self.rank}, num_inducing={self.num_inducing}")
        
        # Check inducing points shape
        inducing_shape = self.model.variational_strategy.base_variational_strategy.inducing_points.shape
        expected_inducing_shape = (self.rank, self.num_inducing, self.train_X.shape[1])
        logger.info(f"Inducing points shape: {inducing_shape}, expected: {expected_inducing_shape}")
        
        if inducing_shape != expected_inducing_shape:
            raise RuntimeError(f"Inducing points shape mismatch: got {inducing_shape}, expected {expected_inducing_shape}")
        
        # Check variational distribution shape
        var_dist = self.model.variational_strategy.base_variational_strategy.variational_distribution
        logger.info(f"Variational distribution batch shape: {var_dist.batch_shape}")
        
        best_loss = float('inf')
        patience_counter = 0
        loss_history = []
        
        for epoch in tqdm(range(epochs), desc="training..."):
            optimizer.zero_grad()
            output = self.model(self.train_X)
            
            beta = self._get_beta(epoch)
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.train_X.shape[0], beta=beta)
            loss = -mll(output, self.train_Y)
            
            # Add small regularization for stability
            reg_loss = 0
            for param in self.model.parameters():
                if param.dim() > 1:  # Only regularize weight matrices
                    reg_loss += 1e-6 * torch.norm(param, p=2)
            
            total_loss = loss + reg_loss
            total_loss.backward()
            
            # Adaptive gradient clipping
            max_grad_norm = 1.0 if epoch < 100 else 0.5
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.likelihood.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            loss_history.append(loss.item())
            
            if epoch % 25 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}, Loss: {loss.item():.3f}, Beta: {beta:.4f}, LR: {current_lr:.6f}")
                if epoch % 100 == 0 and epoch > 0:
                    with torch.no_grad():
                        lengthscales = self.model.covar_module.base_kernel.lengthscale.mean(dim=0)
                        outputscales = self.model.covar_module.outputscale.mean()
                        print(f"  Avg lengthscale: {lengthscales.mean():.3f}, Avg outputscale: {outputscales:.3f}")
            
            # Early stopping
            if epoch > 50:
                recent_losses = loss_history[-20:]
                if len(recent_losses) >= 20:
                    recent_trend = np.polyfit(range(20), recent_losses, 1)[0]  # Linear trend
                    if recent_trend > -1e-5:  # Very small improvement threshold
                        patience_counter += 1
                    else:
                        patience_counter = 0
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                
            if patience_counter > 75:  # More patience for geophysical data
                print(f"Early stopping at epoch {epoch} (trend-based)")
                break
    
    def train(self, train_X, train_Y, epochs=500, lr=0.1, enable_scheduler=True):
        # Convert to torch tensor and move to device if necessary
        if isinstance(train_X, np.ndarray):
            self.train_X = torch.from_numpy(train_X).to(torch.float32)
        else:
            self.train_X = train_X
        if isinstance(train_Y, np.ndarray):
            self.train_Y = torch.from_numpy(train_Y).to(torch.float32)
        else:
            self.train_Y = train_Y
        self.train_X, self.train_Y = self.train_X.to(self.device), self.train_Y.to(self.device)
        
        # Data normalization
        self.train_X, self.train_Y = self._normalize_data(self.train_X, self.train_Y)
        input_dim = self.train_X.shape[1]
        output_dim = self.train_Y.shape[1]
        
        # Determine model parameters
        self._determine_model_parameters(self.train_Y, input_dim, output_dim)
        
        # Initialize inducing points
        inducing_points = self._initialize_inducing_points()
        
        # Setup model
        self._setup_model(input_dim, output_dim, inducing_points)
        
        # Initialize all parameters
        self._initialize_kernel_parameters(input_dim)
        self._initialize_outputscale()
        self._initialize_mean()
        
        # Setup likelihood and noise
        self._initialize_likelihood(output_dim)
        self._initialize_noise(output_dim)
        
        # Set training mode
        self.model.train()
        self.likelihood.train()
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(epochs, lr, enable_scheduler)
        
        # Run training loop
        start_time = time.time()
        self._training_loop(optimizer, scheduler, epochs=epochs)
        training_time = time.time() - start_time
        logger.info(f"Training time: {training_time:.3f} seconds")
        return training_time
    
    def predict(self, test_X: np.ndarray | torch.Tensor):
        # Convert to torch tensor and move to device if necessary
        if isinstance(test_X, np.ndarray):
            self.test_X = torch.from_numpy(test_X).to(torch.float32)
        else:
            self.test_X = test_X
        self.test_X = self.test_X.to(self.device)
        
        # Normalize test data
        self.test_X = (self.test_X - self.train_X_mean) / self.train_X_std
        
        # Evaluation with uncertainty quantification
        logger.info("Starting evaluation with uncertainty quantification...")
        self.model.eval()
        self.likelihood.eval()
        
        start_time = time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():        
            predictions = self.likelihood(self.model(self.test_X))
        infer_time = time.time() - start_time
        logger.info(f"Inference time: {infer_time:.3f} seconds")
        
        # Move to CPU before converting to numpy
        mean = predictions.mean.cpu().detach().numpy()
        std = predictions.stddev.cpu().detach().numpy()
        lower, upper = predictions.confidence_region()
        lower = lower.cpu().detach().numpy()
        upper = upper.cpu().detach().numpy()
        
        # Denormalize (move normalization tensors to CPU for numpy operations)
        pred_mean = mean * self.train_Y_std.cpu().detach().numpy() + self.train_Y_mean.cpu().detach().numpy()
        pred_std = std * self.train_Y_std.cpu().detach().numpy()
        lower = lower * self.train_Y_std.cpu().detach().numpy() + self.train_Y_mean.cpu().detach().numpy()
        upper = upper * self.train_Y_std.cpu().detach().numpy() + self.train_Y_mean.cpu().detach().numpy()
        return pred_mean, pred_std, lower, upper, infer_time
    
    
