import torch
import torch.nn as nn

# VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=None, device='cpu'):
        super(VAE, self).__init__()
        self.device = device
        
        # Default hidden dimensions if none provided
        if hidden_dims is None:
            hidden_dims = [4096, 2048, 1024, 512, 256]
            
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent mean and variance layers
        self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Initialize mean and logvar layers carefully
        nn.init.zeros_(self.mean_layer.bias)
        nn.init.zeros_(self.logvar_layer.bias)
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
        nn.init.xavier_uniform_(self.logvar_layer.weight, gain=0.01)
        
        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
     
    def encode(self, x):
        # encoder
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        # reparameterization
        epsilon = torch.randn_like(logvar).to(self.device)      
        z = mean + torch.exp(0.5 * logvar) * epsilon  
        return z, mean, logvar

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mean, logvar 