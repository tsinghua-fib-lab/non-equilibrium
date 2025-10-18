import os
import numpy as np
import sdeint
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def set_cpu_num(cpu_num: int = 1):
    if cpu_num <= 0: return
    
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
set_cpu_num(1)


def potential(t, x, y, s, term2_varing=True):
    """Calculates the potential V(t, x, y) based on the correct formula from the image."""
    r_sq = x**2 + y**2
    r = np.sqrt(r_sq)

    # Term 1: Angular and rotating part
    arg1 = s * np.arctan2(y, x) - (math.pi/2) * t
    term1 = np.cos(arg1)

    # Term 2: Radial part with time-dependent center
    if term2_varing:
        radial_term_inside = r - 0.5 - 0.5 * np.sin(2 * math.pi * t)
    else:
        radial_term_inside = r - 0.5
    term2 = 10 * radial_term_inside**2

    return term1 + term2

def drift_func(y_flat, t, beta, s, N, term2_varing=True):
    """Drift function for the time-dependent quintuple well SDE."""
    
    # Reshape the flattened input to (N, 2) for easier processing
    y_reshaped = y_flat.reshape(N, 2)
    x = y_reshaped[:, 0]  # x coordinates for all N particles, shape (N,)
    y_vals = y_reshaped[:, 1] # y coordinates for all N particles, shape (N,)

    r_sq = x**2 + y_vals**2 # Squared radial distance for each particle, shape (N,)
    r = np.sqrt(r_sq) # Radial distance for each particle, shape (N,)

    # Add small epsilon for numerical stability near the origin for each particle
    r_sq_safe = np.where(r_sq > 1e-12, r_sq, 1e-12)
    r_safe = np.where(r > 1e-6, r, 1e-6) # Avoid division by zero for r

    # Gradient of the first term (angular part) for each particle
    arg1 = s * np.arctan2(y_vals, x) - (math.pi/2) * t # shape (N,)
    sin_arg1 = np.sin(arg1) # shape (N,)

    # dV1/dx for each particle (shape (N,))
    dV1_dx = s * y_vals * sin_arg1 / r_sq_safe
    # dV1/dy for each particle (shape (N,))
    dV1_dy = -s * x * sin_arg1 / r_sq_safe

    # Gradient of the second term: 10*(sqrt(x^2+y^2) - 3/2 - 1/2*sin(2*pi*t))^2
    # Let R = sqrt(x^2+y^2) - 3/2 - 1/2*sin(2*pi*t)
    # d(10 R^2)/dx = 20 R * dR/dx = 20 R * (x / sqrt(x^2+y^2))
    # d(10 R^2)/dy = 20 R * dR/dy = 20 R * (y / sqrt(x^2+y^2))
    if term2_varing:
        R_term = r - 0.5 - 0.5 * np.sin(2 * math.pi * t) # shape (N,)
    else:
        R_term = r - 0.5

    # dV2/dx for each particle (shape (N,))
    dV2_dx = 20 * R_term * (x / r_safe)
    # dV2/dy for each particle (shape (N,))
    dV2_dy = 20 * R_term * (y_vals / r_safe)

    # Total gradient components for each particle
    dV_dx = dV1_dx + dV2_dx # shape (N,)
    dV_dy = dV1_dy + dV2_dy # shape (N,)

    # The drift is -gradient(V). Stack the x and y components and flatten.
    drift_vector_reshaped = np.stack([-dV_dx, -dV_dy], axis=1) # shape (N, 2)
    return drift_vector_reshaped.ravel() # Flatten to shape (2*N,) for sdeint

def diffusion_func(y_flat, t, beta, N):
    """Diffusion function for the time-dependent quintuple well SDE."""
    # G G^T = 2 * beta^-1 * I
    # G = sqrt(2 / beta) * I
    # The diffusion matrix is diagonal, sigma * Identity matrix of size 2N x 2N
    sigma = np.sqrt(2.0 / beta)
    return np.eye(2 * N) * sigma

def simulate_and_potential(tspan, y0, s, beta=5.0, grid_range=3.0, grid_resolution=(100, 100), term2_varing=True):
    """
    Simulates trajectories for multiple particles and calculates potential well shape at each time step.

    Args:
        tspan: 1D array of time points.
        y0: Initial states, numpy array of shape (N, 2).
        s: Parameter in the potential function.
        beta: Inverse temperature parameter.
        grid_range: The range for x and y coordinates for the potential grid.
        grid_resolution: A tuple (H, W) for the grid resolution.

    Returns:
        A tuple:
            - trajectories: numpy array of shape (len(tspan), N, 2)
            - potentials: numpy array of shape (len(tspan), H, W)
    """
    N = y0.shape[0] # Number of particles
    y0_flat = y0.ravel() # Flatten initial states for sdeint (shape 2*N,)

    # Define lambda functions that pass the number of particles N to the drift and diffusion functions
    f = lambda y_flat, t: drift_func(y_flat, t, beta, s, N, term2_varing=term2_varing)
    G = lambda y_flat, t: diffusion_func(y_flat, t, beta, N)

    # Solve the SDE for the combined system state vector (2*N dimensions)
    trajectory_flat = sdeint.itoint(f, G, y0_flat, tspan) # shape (T, 2*N)

    # Reshape the trajectory back to (T, N, 2)
    trajectories = trajectory_flat.reshape(len(tspan), N, 2)

    # Calculate potential at each time step (this part is independent of the number of particles)
    H, W = grid_resolution
    x_grid = np.linspace(-grid_range, grid_range, W)
    y_grid = np.linspace(-grid_range, grid_range, H)
    X, Y = np.meshgrid(x_grid, y_grid)

    num_time_steps = len(tspan)
    potentials = np.zeros((num_time_steps, H, W))

    # Calculate potential for the entire grid at each time step
    for i, t in enumerate(tspan):
        potentials[i] = potential(t, X, Y, s, term2_varing=term2_varing)

    return trajectories, potentials

def animate_simulation(trajectory, potentials, tspan, grid_range, grid_resolution, num_contour_levels=15):
    """
    Creates and saves a GIF animation of the particle trajectory on the changing potential with clear contour lines.

    Args:
        trajectory: numpy array of shape (len(tspan), N, 2).
        potentials: numpy array of shape (len(tspan), H, W).
        tspan: 1D array of time points.
        grid_range: The range for x and y coordinates for the potential grid.
        grid_resolution: A tuple (H, W) for the grid resolution.
        num_contour_levels: Number of contour levels to display.
    """
    H, W = grid_resolution
    T = len(tspan)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Create grid for contour plots
    x_grid = np.linspace(-grid_range, grid_range, W)
    y_grid = np.linspace(-grid_range, grid_range, H)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Define contour levels based on the overall potential range
    levels = np.linspace(np.min(potentials), np.max(potentials), num_contour_levels)

    def update(i):
        ax.cla() # Clear the axes for redrawing

        # Draw filled contours
        ax.contourf(X, Y, potentials[i], levels=levels, cmap='viridis')
        # Draw contour lines
        ax.contour(X, Y, potentials[i], levels=levels, colors='black', linewidths=0.5)

        # Draw the particle position
        ax.scatter(trajectory[i, :, 0], trajectory[i, :, 1], color='red', s=10)

        # Update title and restore axis properties
        ax.set_title(f'Time: {tspan[i]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([-grid_range, grid_range])
        ax.set_ylim([-grid_range, grid_range])
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()

        # Return empty list when blit=False
        return []

    # Create animation (blit=False because we clear and redraw the axes)
    ani = animation.FuncAnimation(fig, update, frames=T, blit=False, interval=50) # interval in ms

    print("Saving animation...")
    # Use pillow writer, requires 'pillow' to be installed (`pip install pillow`)
    ani.save(f'data/N{num_particles}.gif', writer='pillow', fps=20) # fps: frames per second
    print("Animation saved as simulate.gif")

    plt.close(fig) # Close the figure after saving


if __name__ == '__main__':
    beta_val = 10.0
    s_val = 5
    num_particles = 1000
    term2_varing=False
    tspan_val = np.linspace(0.0, 4.0 * s_val, 1000)

    grid_range_val = 1.0
    y0_val = np.random.uniform(low=-grid_range_val, high=grid_range_val, size=[num_particles, 2])
    grid_resolution_val = (100, 100)

    try:
        data = np.load(f'data/N{num_particles}_s{s_val}.npz')
        trajectory_result = data['trajectory']
        potentials_result = data['potential']
    except:
        batch_size = 10
        if num_particles <= batch_size:
            trajectory_result, potentials_result = simulate_and_potential(
                tspan_val, y0_val, s_val, grid_range=grid_range_val, grid_resolution=grid_resolution_val, term2_varing=term2_varing
            )
        else:
            batched_trajectories = []

            from tqdm import tqdm
            for start_idx in tqdm(range(0, num_particles, batch_size), total=num_particles//batch_size):
                end_idx = min(start_idx + batch_size, num_particles)
                y0_batch = y0_val[start_idx:end_idx]

                traj_batch, potentials_result = simulate_and_potential(
                    tspan_val, y0_batch, s_val, grid_range=grid_range_val, grid_resolution=grid_resolution_val, term2_varing=term2_varing
                )

                batched_trajectories.append(traj_batch)

            trajectory_result = np.concatenate(batched_trajectories, axis=1)



        print("Trajectory shape:", trajectory_result.shape)
        print("Potentials shape:", potentials_result.shape)
        os.makedirs('data/', exist_ok=True)
        np.savez(f'data/N{num_particles}_s{s_val}.npz', trajectory=trajectory_result, potential=potentials_result)
    
    # Create and save the animation
    animate_simulation(trajectory_result[::10], potentials_result[::10], tspan_val[::10], grid_range_val, grid_resolution_val)