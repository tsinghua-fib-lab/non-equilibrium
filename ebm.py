import os
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, math
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from torch.utils.data import Dataset, DataLoader, TensorDataset # Only these needed for dataset/dataloader
import os # For data loading
import warnings # For KDE warnings

# Ignore KDE warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- Helpers ---
# Fourier Features encoding (assuming it takes (..., 1) -> (..., num_encodings * 2))
def get_fourier_features(x, num_encodings):
    if x.dim() == 1: x = x.unsqueeze(-1)
    scales = 2.**torch.arange(num_encodings, device=x.device) * math.pi
    x_scaled = x * scales; x_scaled = x_scaled.view(*x.shape[:-1], -1)
    return torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1)

# Get list of KDE functions (from previous turn)
def get_kde_functions(trajectories):
    T, N, _ = trajectories.shape
    kdes = [] # List to store KDE objects
    for i in range(T):
        pos = trajectories[i] # (N, 2)
        if pos.shape[0] > 1:
            try: kdes.append(gaussian_kde(pos.T)) # Compute KDE (input needs (ndim, nsamples))
            except np.linalg.LinAlgError: kdes.append(None) # Handle failure
        else: kdes.append(None) # Not enough points
    return kdes


# --- Model (Time Dependent Energy Field - MLP(x, t)) ---
class EnergyField(nn.Module):
    """
    Parameterizes a time-dependent energy field E(x, t) using an MLP.
    Input: x (..., 2) spatial coords, t (..., 1) dynamic time.
    Output: (..., 1) scalar energy.
    """
    # sd: spatial_dim (2), td: time_dim (1), ffs: fourier_features_spatial, 
    # fft: fourier_features_time, hs: hidden_size, nl: num_layers
    def __init__(self, sd=2, td=1, ffs=64, fft=64, hs=256, nl=4): 
        super().__init__()
        self.sd, self.td = sd, td
        self.ffs, self.fft = ffs, fft
        
        # Scaling factor for time input (adjust based on max dynamic time in data)
        self.max_td = 20.0 # Example max time

        # Calculate output dimensions after Fourier encoding
        self.sfo_dim = sd * ffs * 2 
        self.tfo_dim = td * fft * 2 

        # MLP layers
        inp_dim = self.sfo_dim + self.tfo_dim
        
        layers = []
        layers.append(nn.Linear(inp_dim, hs)); layers.append(nn.SiLU()) 
        for _ in range(nl - 2): layers.append(nn.Linear(hs, hs)); layers.append(nn.SiLU()) 
        layers.append(nn.Linear(hs, 1)) # Output: scalar energy

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):        
        # Encode spatial coordinates element-wise
        x_encoded_list = []
        for i in range(self.sd):
             x_encoded_list.append(get_fourier_features(x[..., i].unsqueeze(-1), self.ffs))
        x_encoded = torch.cat(x_encoded_list, dim=-1) # Shape (..., sfo_dim)

        # Encode time (scale and ensure shape is (..., 1))
        t_scaled = t.float() / self.max_td 
        t_encoded = get_fourier_features(t_scaled, self.fft)

        # Concatenate encoded features
        combined_features = torch.cat([x_encoded, t_encoded], dim=-1) 
        energy = self.mlp(combined_features) # Output shape (..., 1)

        return energy 

# --- Data Loading and Prep for Training ---
def load_data_prep_train_grid(n_p=350, device="cpu", train_grid_range=10.0, train_grid_res=(100, 100)):
    try: d = np.load(f'data/N{n_p}_s5.npz')
    except FileNotFoundError: exit("Data not found.")

    traj_full_np = d['trajectory'] # Shape (T_sim_full, N_sim, 2)
    T_sim_full, N_sim, _ = traj_full_np.shape

    # Derive the FULL sequence of original dynamic times
    sim_dynamic_times_full = np.linspace(0, 4.0*5, T_sim_full, dtype=np.float32) # (T_sim_full,)

    # Compute KDEs for all time steps
    print("Computing KDE functions...")
    kde_funcs_list = get_kde_functions(traj_full_np)
    print(f"KDE functions computed for {len(kde_funcs_list)} time steps.")

    # --- Pre-calculate Target Energy on a TRAINING GRID ---
    Hs_train, Ws_train = train_grid_res
    # Create training grid points (2, Hs_train*Ws_train)
    x_grid_train = np.linspace(-train_grid_range, train_grid_range, Ws_train) 
    y_grid_train = np.linspace(-train_grid_range, train_grid_range, Hs_train)
    grid_x_train, grid_y_train = np.meshgrid(x_grid_train, y_grid_train) 
    grid_points_flat_train = np.vstack([grid_x_train.ravel(), grid_y_train.ravel()]) # Shape (2, Hs_train*Ws_train)

    # Initialize target energy grid with a high value (low density)
    high_energy_target = 1000.0 # Represents very low probability
    target_kde_energy_grid_np = np.full((T_sim_full, Hs_train, Ws_train), high_energy_target, dtype=np.float32) 

    print(f"Pre-calculating target KDE energy on a {Hs_train}x{Ws_train} grid...")
    # Loop through all time steps
    for i in range(T_sim_full):
        kde_func = kde_funcs_list[i]
        # Evaluate KDE density on the grid points
        density_values = kde_func.evaluate(grid_points_flat_train) # Shape (Hs_train*Ws_train,)

        # Convert density to energy: -log(density)
        positive_density_mask = density_values > 1e-10 # Mask where density is significantly positive
        energy_values = np.full_like(density_values, high_energy_target) # Initialize with high energy
        
        # Calculate energy only for points with positive density
        energy_values[positive_density_mask] = -np.log(density_values[positive_density_mask]) 

        # Clamp energy values to a maximum (preventing infinity/very large numbers)
        energy_values[energy_values > high_energy_target] = high_energy_target 

        # Reshape and store the energy values
        target_kde_energy_grid_np[i] = energy_values.reshape(Hs_train, Ws_train)

    print("Pre-calculation of grid targets done.")

    grid_points_flat_all_times = np.tile(grid_points_flat_train.T, (T_sim_full, 1)).reshape(-1, 2) # Repeat grid points for all times (Total_grid_points, 2)
    time_indices_flat_all_times = np.repeat(np.arange(T_sim_full), Hs_train * Ws_train) # Repeat time indices for all grid points (Total_grid_points,)
    target_energy_flat_all_times = target_kde_energy_grid_np.reshape(-1, 1) # (Total_grid_points, 1)

    grid_points_t = torch.tensor(grid_points_flat_all_times, dtype=torch.float).to(device)
    time_indices_t = torch.tensor(time_indices_flat_all_times, dtype=torch.long).to(device) 
    target_energy_t = torch.tensor(target_energy_flat_all_times, dtype=torch.float).to(device) 

    dataset = TensorDataset(grid_points_t, time_indices_t, target_energy_t)

    return dataset, sim_dynamic_times_full, traj_full_np


# --- Training ---
def train_energy_field_grid(model, dataset, original_times, epochs, lr, bs, N_sim, device): 
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # Loss for energy regression

    print(f"Train EBM ({len(dataset)} samples, bs {bs}) on {device} using pre-calculated GRID targets...")

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        # DataLoader yields (grid_point, time_index, target_energy)
        for grid_pos_batch, time_idx_batch, target_energy_batch in dataloader:
            t_dyn_batch = torch.tensor(original_times[time_idx_batch.cpu().numpy()], dtype=torch.float).to(device).unsqueeze(-1) # (bs, 1)
            E_pred = model(grid_pos_batch, t_dyn_batch) # (bs, 1)
            loss = loss_fn(E_pred, target_energy_batch)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * grid_pos_batch.size(0) # Accumulate loss

        avg_l = loss_sum / len(dataset) # Average loss per sample

        if (epoch + 1) % 1 == 0:
            print(f"E {epoch+1}/{epochs}, Loss: {avg_l:.5f}")
    
    os.makedirs(f'log/N{N_sim}_s5/ebm', exist_ok=True)
    torch.save(model.state_dict(), f'log/N{N_sim}_s5/ebm/energy.pt')    
    print("Train done.")
    return model


# --- Testing/Inference ---
def evaluate_field_and_generate_particles(model, time_points, grid_range, grid_resolution, num_gen_particles, device):
    model.eval() # Set model to evaluation mode
    Hs, Ws = grid_resolution
    T_eval = len(time_points) # Number of time points to evaluate
    
    x_grid = np.linspace(-grid_range, grid_range, Ws) # x coordinates of grid lines
    y_grid = np.linspace(-grid_range, grid_range, Hs) # y coordinates of grid lines
    grid_x, grid_y = np.meshgrid(x_grid, y_grid) # Meshgrid for grid coordinates
    grid_points_flat_np = np.vstack([grid_x.ravel(), grid_y.ravel()]).T # Flatten grid points to (Hs*Ws, 2)
    grid_points_t = torch.tensor(grid_points_flat_np, dtype=torch.float).to(device) # Convert to tensor for device

    dx = (x_grid[1] - x_grid[0]) if Ws > 1 else (grid_range * 2 / Ws) # Width of a single grid cell in x-direction
    dy = (y_grid[1] - y_grid[0]) if Hs > 1 else (grid_range * 2 / Hs) # Height of a single grid cell in y-direction

    energy_grid_results = torch.zeros(T_eval, Hs, Ws, device=device) # To store energy grids
    generated_particles_results = torch.zeros(T_eval, num_gen_particles, 2, device=device) # To store generated particle coordinates

    with torch.no_grad(): # Disable gradient calculation during evaluation
        for i, t_val in enumerate(time_points):
            t_batch_grid = torch.full((Hs*Ws, 1), float(t_val), device=device) 

            energy_grid_flat = model(grid_points_t, t_batch_grid) # Output shape (Hs*Ws, 1)
            energy_grid_results[i] = energy_grid_flat.reshape(Hs, Ws) # Store energy grid (H, W)

            # --- Generate Particles based on this energy field ---
            unnorm_probs_flat = torch.exp(-energy_grid_flat) # Shape (Hs*Ws, 1)
            
            pmf_flat = unnorm_probs_flat.squeeze(-1) # Shape (Hs*Ws,)
            
            if pmf_flat.sum() < 1e-10: 
                pmf_flat = torch.ones_like(pmf_flat) 
            pmf_flat /= pmf_flat.sum() # Normalize to sum to 1

            sampled_flat_indices = torch.multinomial(pmf_flat, num_gen_particles, replacement=True) # (num_gen_particles,)

            sampled_row_indices, sampled_col_indices = np.unravel_index(
                sampled_flat_indices.cpu().numpy(), (Hs, Ws)
            )
            
            x_corners = torch.tensor(x_grid[sampled_col_indices], dtype=torch.float, device=device)
            y_corners = torch.tensor(y_grid[sampled_row_indices], dtype=torch.float, device=device)

            random_offsets_x = torch.rand(num_gen_particles, device=device) * dx
            random_offsets_y = torch.rand(num_gen_particles, device=device) * dy

            generated_x = x_corners + random_offsets_x
            generated_y = y_corners + random_offsets_y
            generated_particles_t = torch.stack([generated_x, generated_y], dim=-1) # Shape (num_gen_particles, 2)

            generated_particles_results[i] = generated_particles_t # Shape (num_gen_particles, 2)

    return energy_grid_results.cpu().numpy(), generated_particles_results.cpu().numpy()



# --- Main Execution ---
if __name__ == '__main__':
    # Hyperparameters
    epochs = 50
    lr = 1e-3
    bs = 512
    device = "cuda:3"
    n_p_sim = 1000

    # Training grid parameters
    train_grid_range_val = 1.0
    train_grid_res_val = (50, 50) 

    # Viz/Eval parameters
    eval_grid_range_val = train_grid_range_val
    eval_grid_res_val = train_grid_res_val
    eval_time_points = np.linspace(0, 20, 1000)

    # 1. Load data, pre-calculate KDE energy targets on training grid, and prep dataset
    dataset, original_times, traj_full_np = load_data_prep_train_grid(
        n_p=n_p_sim, device=device, train_grid_range=train_grid_range_val, train_grid_res=train_grid_res_val
    ) 
    
    try:
        trained_model = EnergyField().to(device) # Assuming EnergyField is defined with sd=2 default
        trained_model.load_state_dict(torch.load(f'log/N{n_p_sim}_s5/ebm/energy.pt', map_location=device))
        d = np.load(f'data/N{n_p_sim}_s5.npz')
        traj_full_np = d['trajectory'] # (T_sim_full, N_sim, 2)
    except:
        # 2. Model, Opt
        model = EnergyField(sd=2).to(device) # Assuming EnergyField is defined with sd=2 default
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        # 3. Train Energy Field Model using pre-calculated KDEs on the grid
        print("\n--- Training Energy Field with Pre-calculated GRID Targets ---")
        trained_model = train_energy_field_grid(
            model, dataset, original_times, epochs, lr, bs, n_p_sim, device
        )

    # 4. Test/Inference
    print("\n--- Evaluating Energy Field and Density ---")
    energy_grids_np, generated_particles_np = evaluate_field_and_generate_particles(trained_model, eval_time_points, eval_grid_range_val, eval_grid_res_val, n_p_sim, device)
    np.savez(f'log/N{n_p_sim}_s5/ebm/result.npz', energy_grids_np=energy_grids_np, generated_particles_np=generated_particles_np)
    
    # --- Example Visualization of Energy Field and Particle Densities ---
    eval_idx = 2
    grid_range = eval_grid_range_val
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(energy_grids_np[eval_idx], origin='lower', extent=[-grid_range, grid_range, -grid_range, grid_range], cmap='viridis')
    plt.colorbar(label='Energy')
    plt.title(f'Energy Field at t={eval_time_points[eval_idx]:.2f}')
    plt.xlabel('x'); plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(1, 2, 2)
    generated_particles_at_eval_t = generated_particles_np[np.searchsorted(original_times, eval_time_points[eval_idx])]
    scatter = plt.scatter(generated_particles_at_eval_t[:, 0], generated_particles_at_eval_t[:, 1], s=10, alpha=0.8)
    plt.title(f'Generated particles at t={eval_time_points[eval_idx]:.2f}')
    plt.xlabel('x'); plt.ylabel('y')
    plt.xlim([-grid_range, grid_range]); plt.ylim([-grid_range, grid_range])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    os.makedirs(f'log/N{n_p_sim}_s5/ebm', exist_ok=True)
    plt.savefig(f'log/N{n_p_sim}_s5/ebm/ebm_N{n_p_sim}.png', dpi=300)