import torch
import torch.nn as nn
import math

# Assumed get_fourier_features is defined elsewhere and available
# from your_model_file import get_fourier_features # Example import

def get_fourier_features(x, num_encodings=32):
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    scales = 2.**torch.arange(num_encodings, device=x.device) * math.pi # (num_encodings,)
    x_scaled = x * scales # (..., 1, num_encodings) -> (..., num_encodings) after broadcasting
    x_scaled = x_scaled.view(*x.shape[:-1], -1) # (..., num_encodings)
    fourier_features = torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1) # (..., num_encodings * 2)

    return fourier_features


class NoiseModel(nn.Module):
    # id: input_dim (2), f: fourier_features (32), hs: hidden_size (128), Tn: Total diffusion timesteps (1000)
    def __init__(self, input_dim=2, f=32, hs=128, N_max=1000, T_max=20.0): 
        super().__init__()
        self.input_dim, self.T_max, self.N_max = input_dim, T_max, N_max
        self.ff_dim = f * 2 # Dimension after Fourier encoding
        # MLP layers: input is concatenated noisy data (x) and time features
        self.l1 = nn.Linear(input_dim + self.ff_dim * 2, hs) 
        self.l2 = nn.Linear(hs, hs)
        self.l3 = nn.Linear(hs, input_dim) # Output predicts epsilon (same dim as x)
        self.relu = nn.SiLU() # Activation function

    # Forward pass: x is noisy data, t is current timestep (float, 0 to Tn-1)
    def forward(self, x, t, n):
        # Scale timestep to 0-1 range before Fourier encoding
        n_scaled = n / self.N_max 
        n_f = get_fourier_features(n_scaled.unsqueeze(-1), self.ff_dim // 2) # Get Fourier features (needs (..., 1) input)
        t_scaled = t / self.T_max
        t_f = get_fourier_features(t_scaled.unsqueeze(-1), self.ff_dim // 2) # Get Fourier features (needs (..., 1) input)
        
        # Concatenate noisy data and time features
        x_t = torch.cat([x, t_f, n_f], dim=-1)
        
        # Pass through MLP layers with activation
        h = self.relu(self.l1(x_t))
        h = self.relu(self.l2(h))
        
        # Output predicts noise epsilon
        return self.l3(h)
