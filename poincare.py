import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpmath import mp

df = pd.read_csv('./Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv', dtype=str) 

def count_decimals(value):
    if '.' in value:
        return len(value.split('.')[1])
    return 0

# Find the maximum number of decimal places
max_decimal_places = 0
for col in ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']:
    max_decimal_places = max(max_decimal_places, df[col].apply(count_decimals).max())

# Set the global decimal places
mp.dps = max_decimal_places
mp.prec = 3.33 * max_decimal_places  # Set precision to 3.33 times the decimal places

# takes input value (expected to be a string) and converts it to mpf object
def string_to_mpf(value):
    return mp.mpf(value)

for col in ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']:
    df[col] = df[col].apply(string_to_mpf)

# convert to numeric
df['Timestep'] = pd.to_numeric(df['Timestep'])

# Separate forward and backward trajectories
forward_trajectory = df[df['Phase'].astype(int) == 1]
backward_trajectory = df[df['Phase'].astype(int) == -1]

# Get unique timesteps and calculate midpoint
total_timesteps = len(df['Timestep'].unique())
midpoint = total_timesteps // 2  # Midpoint corresponds to t=T=lifetime
timesteps = df['Timestep'].unique()


# function to generate poincaré map
# define a surface of section
def find_poincare_intersections(trajectory, surface_col='Z Position', velocity_col='Z Velocity'):
    poincare_points = []

    # Group by particle
    for particle_id, particle_data in trajectory.groupby('Particle Number'):
        particle_data = particle_data.sort_values('Timestep')

        # Iterate over timesteps to detect crossings
        positions = particle_data[surface_col].values
        velocities = particle_data[velocity_col].values
        timesteps = particle_data['Timestep'].values
        x_positions, y_positions = particle_data['X Position'].values, particle_data['Y Position'].values
        x_velocities, y_velocities = particle_data['X Velocity'].values, particle_data['Y Velocity'].values


        for i in range(len(positions) - 1):
            # to detect crossing: the position changes sign and the velocity is positive
            if positions[i] * positions[i + 1] < 0 and velocities[i] > 0:  # Z = 0 and Vz > 0
                # linear interpolation to find the next crossing point
                t_cross = (timesteps[i] - positions[i] * (timesteps[i + 1] - timesteps[i]) / (positions[i + 1] - positions[i]))
                x_cross = x_positions[i] + (x_positions[i + 1] - x_positions[i]) * (t_cross - timesteps[i]) / (timesteps[i + 1] - timesteps[i])
                y_cross = y_positions[i] + (y_positions[i + 1] - y_positions[i]) * (t_cross - timesteps[i]) / (timesteps[i + 1] - timesteps[i])
                vx_cross = x_velocities[i] + (x_velocities[i + 1] - x_velocities[i]) * (t_cross - timesteps[i]) / (timesteps[i + 1] - timesteps[i])
                vy_cross = y_velocities[i] + (y_velocities[i + 1] - y_velocities[i]) * (t_cross - timesteps[i]) / (timesteps[i + 1] - timesteps[i])
                poincare_points.append([particle_id, x_cross, vx_cross, y_cross, vy_cross])
    
    return np.array(poincare_points)

# use only forward trajectory
poincare_points = find_poincare_intersections(forward_trajectory)

# Plot Poincaré map for all particles (X components)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for particle_id in np.unique(poincare_points[:, 0]):
    particle_points = poincare_points[poincare_points[:, 0] == particle_id]
    plt.scatter(particle_points[:, 1], particle_points[:, 2], s=5, label=f'Particle {particle_id}')
plt.xlabel('X Position')
plt.ylabel('X Velocity')
plt.title('Poincaré Map (X Components)')
plt.legend()
plt.grid()

# Plot Poincaré map for all particles (Y components)
plt.subplot(1, 2, 2)
for particle_id in np.unique(poincare_points[:, 0]):
    particle_points = poincare_points[poincare_points[:, 0] == particle_id]
    plt.scatter(particle_points[:, 3], particle_points[:, 4], s=5, label=f'Particle {particle_id}')
plt.xlabel('Y Position')
plt.ylabel('Y Velocity')
plt.title('Poincaré Map (Y Components)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

""" 

using the map proposed here: https://www.researchgate.net/figure/Poincare-sections-in-the-planar-circular-restricted-three-body-problem-corresponding-to-a_fig2_250082571

def find_poincare_intersections(trajectory, surface_col='X Position', velocity_col='X Velocity', y_col='Y Position', y_vel_col='Y Velocity'):
    poincare_points = []
    mu = 0.5
    x_surface = 1 - mu

    # Group by particle
    for particle_id, particle_data in trajectory.groupby('Particle Number'):
        particle_data = particle_data.sort_values('Timestep')

        # Convert columns to numeric values
        x_positions = particle_data[surface_col].values
        x_velocities = particle_data[velocity_col].values
        y_positions = particle_data[y_col].values
        y_velocities = particle_data[y_vel_col].values
        timesteps = particle_data['Timestep'].values

        # Iterate over timesteps to detect crossings
        for i in range(len(x_positions) - 1):
            # To detect crossing: the position crosses x_surface, y > 0, and x_vel > 0
            if (x_positions[i] - x_surface) * (x_positions[i + 1] - x_surface) < 0 and y_positions[i] > 0 and x_velocities[i] > 0:
                # Linear interpolation to find the crossing point
                t_cross = (timesteps[i] - (x_positions[i] - x_surface) * (timesteps[i + 1] - timesteps[i]) / (x_positions[i + 1] - x_positions[i]))
                y_cross = y_positions[i] + (y_positions[i + 1] - y_positions[i]) * (t_cross - timesteps[i]) / (timesteps[i + 1] - timesteps[i])
                vx_cross = x_velocities[i] + (x_velocities[i + 1] - x_velocities[i]) * (t_cross - timesteps[i]) / (timesteps[i + 1] - timesteps[i])
                vy_cross = y_velocities[i] + (y_velocities[i + 1] - y_velocities[i]) * (t_cross - timesteps[i]) / (timesteps[i + 1] - timesteps[i])
                poincare_points.append([particle_id, x_surface, vx_cross, y_cross, vy_cross])
    
    return np.array(poincare_points) """