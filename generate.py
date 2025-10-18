import os
import torch
import numpy as np
from tqdm import tqdm
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import NoiseModel


# --- Concise Generation Function ---
@torch.no_grad() # Disable gradient calculation
def generate_trajectories_concise(model, scheduler, num_particles, dynamic_times, device):
    T, input_dim = len(dynamic_times), model.input_dim
    generated_data = torch.zeros(T, num_particles, input_dim, device=device)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps) # Prepare diffusion timesteps
    scheduler.config.clip_sample = False 

    for i, t_dyn_val in tqdm(enumerate(dynamic_times), total=T):
        latent = torch.randn(num_particles, input_dim, device=device)
        t_dyn_batch = torch.full((num_particles,), t_dyn_val, device=device) # Dynamic time batch (constant for this t_dyn_val)

        for s_diff_val in scheduler.timesteps:
            s_diff_batch = torch.full((num_particles,), float(s_diff_val), device=device) # Diffusion time batch (changes per step)
            model_output = model(latent, t_dyn_batch, s_diff_batch) # Predict noise/score
            latent = scheduler.step(model_output, s_diff_val, latent).prev_sample

        generated_data[i] = latent # Store generated samples for this dynamic time

    return dynamic_times.cpu().numpy(), generated_data.cpu().numpy()


@torch.no_grad()
def sample_and_record_dynamics(model, diffusion_N, scheduler, num_gen_particles, t_dyn_value, grid_range, grid_resolution, device):
    H, W = grid_resolution # Grid dimensions
    input_dim = model.input_dim # Input/output dimension of the model (should be 2)

    # Arrays to store the recorded data
    particle_states_sequence = np.zeros((diffusion_N, num_gen_particles, input_dim), dtype=np.float32) 
    gradient_field_sequence = np.zeros((diffusion_N, H, W, input_dim), dtype=np.float32)

    # Create grid points for evaluating the gradient field
    x_grid = np.linspace(-grid_range, grid_range, W)
    y_grid = np.linspace(-grid_range, grid_range, H)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_points_flat_np = np.vstack([grid_x.ravel(), grid_y.ravel()]).T # Shape (H*W, 2)
    grid_points_t = torch.tensor(grid_points_flat_np, dtype=torch.float).to(device) # Tensor for device (H*W, 2)

    # Prepare dynamic time tensor (constant for the entire diffusion process run)
    t_dyn_batch_particles = torch.full((num_gen_particles,), t_dyn_value, device=device) # Shape (num_gen_particles,)
    t_dyn_batch_grid = torch.full((H*W,), t_dyn_value, device=device) # Shape (H*W,)

    # Set diffusion timesteps for the reverse process (runs from diffusion_N-1 down to 0)
    scheduler.set_timesteps(diffusion_N)
    scheduler.config.clip_sample = False 

    # Start from random noise at the last timestep (diffusion_N-1)
    x_current = torch.randn(num_gen_particles, input_dim, device=device) # State at step diffusion_N-1 (in normalized space)

    # Loop backward through diffusion timesteps (from diffusion_N-1 down to 0)
    for i, s_diff_val in enumerate(scheduler.timesteps):
         # Store particle state AT the start of this step (x_{s_diff_val})
         particle_states_sequence[i] = x_current.cpu().numpy()

         # Prepare diffusion time tensor for the current step (for model input)
         s_diff_batch_particles = torch.full((num_gen_particles,), float(s_diff_val), device=device) 
         s_diff_batch_grid = torch.full((H*W,), float(s_diff_val), device=device) 

         # --- Calculate Gradient Field on Grid at step s_diff_val ---
         epsilon_pred_grid = model(grid_points_t, t_dyn_batch_grid, s_diff_batch_grid) # Output shape (H*W, 2)
         gradient_field_sequence[i] = -epsilon_pred_grid.reshape(H, W, input_dim).cpu().numpy() # Reshape to (H, W, 2)

         # --- Perform Scheduler Step for Particles ---
         epsilon_pred_particles = model(x_current, t_dyn_batch_particles, s_diff_batch_particles) # Output shape (num_gen_particles, 2)
         x_next = scheduler.step(epsilon_pred_particles, s_diff_val, x_current).prev_sample

         x_current = x_next

    return particle_states_sequence, gradient_field_sequence # (diffusion_N, num_gen_particles, 2), (diffusion_N, H, W, 2)


def animate_diffusion_process(particle_traj_diffusion, gradient_fields, dynamic_time_value, diffusion_timesteps_values, grid_range, save_path):
    DN, N_gen, _ = particle_traj_diffusion.shape # DN: Total diffusion steps
    DN_field, H, W, _ = gradient_fields.shape # Should be same as DN
    fig, ax = plt.subplots(figsize=(6, 6)) # Create figure and axes for plotting
    x_grid = np.linspace(-grid_range, grid_range, W)
    y_grid = np.linspace(-grid_range, grid_range, H)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # --- Update function for animation ---
    def update(i):
        ax.cla() # Clear axes from previous frame

        # --- Debug Prints and Checks ---
        diffusion_step_val = diffusion_timesteps_values[i] 
        print(f"Processing frame {i+1}/{DN} (Diff Step: {diffusion_step_val:.0f})...", end="")

        particles_at_step = particle_traj_diffusion[i] # Positions (N_gen, 2)
        field_at_step = gradient_fields[i]       # Field (H, W, 2)
        
        U_field = field_at_step[:, :, 0] # X components
        V_field = field_at_step[:, :, 1] # Y components

        max_magnitude = np.max(np.sqrt(U_field**2 + V_field**2))
        plot_range_dim = grid_range * 2 
        desired_arrow_length_frac = 0.05 
        desired_arrow_length_data_units = plot_range_dim * desired_arrow_length_frac

        scale_val = max_magnitude / desired_arrow_length_data_units if desired_arrow_length_data_units > 0 and max_magnitude > 0 else 1.0
        scale_val = 1.0 if not np.isfinite(scale_val) or scale_val == 0 else scale_val # Ensure scale is finite and non-zero

        Q = ax.quiver(X_grid, Y_grid, U_field, V_field, units='xy', angles='xy', scale=scale_val)

        ax.scatter(particles_at_step[:, 0], particles_at_step[:, 1], color='red', s=10, label='Particles')

        ax.set_title(f'Dyn Time: {dynamic_time_value:.2f}, Diff Step: {diffusion_step_val:.0f} ({i+1}/{DN})') 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([-grid_range, grid_range])
        ax.set_ylim([-grid_range, grid_range])
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        print('Done.')
        return [] # Keep return [] for blit=False

    # --- Create animation ---
    print("Creating animation...")
    ani = animation.FuncAnimation(fig, update, frames=DN, blit=False, interval=50)

    # --- Save animation ---
    print(f"Saving animation to {save_path}...")
    ani.save(save_path, writer='pillow', fps=20) 
    print("Animation saved.")
    plt.close(fig) # Close the figure after saving to free memory




# --- Setup ---
num_particles = 1000
diffusion_N = 1000
grid_range = 1.0
grid_resolution = (20, 20)
device = "cuda:1"

model = NoiseModel().to(device)
scheduler = DDPMScheduler(num_train_timesteps=diffusion_N, prediction_type='epsilon', clip_sample=False)
model.load_state_dict(torch.load(f"log/N{num_particles}_s{5}/diffusion/diffusion.pt", map_location=device))
model.eval()



num_dynamic_time_steps = 1000 # Number of dynamic time points
max_dynamic_time = 4.0 * 5 # Max dynamic time
dynamic_times_for_gen = torch.linspace(0, max_dynamic_time, num_dynamic_time_steps, dtype=torch.float, device=device)
dynamic_times_np, generated_trajectories_np = generate_trajectories_concise(
    model, scheduler, num_particles, dynamic_times_for_gen, device
)
print(f"Generated trajectory shape: {generated_trajectories_np.shape}")
np.savez(f"log/N{num_particles}_s{5}/diffusion/generate.npz", dynamic_times=dynamic_times_np, trajectories=generated_trajectories_np)




# for t_dyn in tqdm(np.linspace(0, 20, 1000), total=1000):
#     recorded_particle_traj_diffusion, recorded_gradient_fields = sample_and_record_dynamics(
#         model, 
#         diffusion_N,
#         scheduler, 
#         num_particles, 
#         t_dyn, 
#         grid_range, 
#         grid_resolution, 
#         device
#     )
#     os.makedirs(f"log/N{num_particles}_s{5}/diffusion/sample/", exist_ok=True)
#     np.savez(f"log/N{num_particles}_s{5}/diffusion/sample/generate_t{t_dyn:.2f}.npz", recorded_particle_traj_diffusion=recorded_particle_traj_diffusion, recorded_gradient_fields=recorded_gradient_fields)


#     diffusion_timesteps_values_np = scheduler.timesteps.cpu().numpy()
#     animate_diffusion_process(
#         recorded_particle_traj_diffusion, # Particle positions sequence
#         recorded_gradient_fields,       # Gradient field sequence
#         t_dyn,             # The constant dynamic time value
#         diffusion_timesteps_values_np, # The sequence of diffusion timestep values (numpy array)
#         grid_range,              # Spatial range used for the grid
#         f"log/N{num_particles}_s{5}/diffusion/generate_t{t_dyn:.2f}.gif"             # Path to save the GIF
#     )