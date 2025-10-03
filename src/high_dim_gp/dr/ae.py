import torch.nn as nn

# AE architecture
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=None):
        super().__init__()
        
        # Default hidden dimensions if none provided
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]
            
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x