import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from mpmath import mp

# Read the CSV file with all values as strings
df = pd.read_csv('./Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv', dtype=str)

# Convert necessary columns to mpmath's mpf type for high precision
for col in ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']:
    df[col] = df[col].apply(mp.mpf)

# Separate forward and backward trajectories
forward_trajectory = df[df['Phase'].astype(int) == 1]
backward_trajectory = df[df['Phase'].astype(int) == -1]

# Get unique timesteps and calculate midpoint
total_timesteps = len(df['Timestep'].unique())
midpoint = total_timesteps // 2  # Midpoint corresponds to t=T=lifetime
timesteps = df['Timestep'].unique()

# Define symmetric timesteps
timesteps_forward = timesteps[:midpoint]
timesteps_backward = timesteps[midpoint+1:][::-1]  # Reverse order for symmetry
print(len(timesteps_forward), len(timesteps_backward))

# Get unique particles
particles = df['Particle Number'].unique()

# Phase space distance between two states
def compute_delta(forward_state, backward_state):
    # Extract positions and convert to mpmath
    x_f = forward_state['X Position'].values
    y_f = forward_state['Y Position'].values
    z_f = forward_state['Z Position'].values
    x_b = backward_state['X Position'].values
    y_b = backward_state['Y Position'].values
    z_b = backward_state['Z Position'].values

    # Extract velocities and convert to mpmath
    vx_f = forward_state['X Velocity'].values
    vy_f = forward_state['Y Velocity'].values
    vz_f = forward_state['Z Velocity'].values
    vx_b = -backward_state['X Velocity'].values
    vy_b = -backward_state['Y Velocity'].values
    vz_b = -backward_state['Z Velocity'].values

    # Compute the phase-space distance for this particle using mpmath
    diff_vel = [(vx_f[i] - vx_b[i])**2 + (vy_f[i] - vy_b[i])**2 + (vz_f[i] - vz_b[i])**2 for i in range(len(vx_f))]
    diff_pos = [(x_f[i] - x_b[i])**2 + (y_f[i] - y_b[i])**2 + (z_f[i] - z_b[i])**2 for i in range(len(x_f))]
    
    return sum(diff_vel) + sum(diff_pos)

# Compute delta at each symmetric timestep
delta_per_step = []

for i in range(len(timesteps_forward)):
    delta_sum = mp.mpf(0)  # Initialize as mp.mpf for high precision
    for p in particles:
        forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
        backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
        
        forward_state = forward_p[forward_p['Timestep'] == timesteps_forward[i]]
        backward_state = backward_p[backward_p['Timestep'] == timesteps_backward[i]]
        
        # Check the comparison - seems correct
        if i == 30 and p == particles[0]:  # Check for a timestep and for a particle
            print("Forward state compared:", forward_state)
            print("Backward state compared:", backward_state)
    
        delta = compute_delta(forward_state, backward_state)
        delta_sum += delta
        
    delta_per_step.append(delta_sum)  # Append the delta summed over the bodies for this timestep

# Compute delta between initial and final states
delta_initial_final = mp.mpf(0)  # Initialize as mp.mpf for high precision

initial_timestep = timesteps[0]  # Start of forward trajectory
final_timestep = timesteps[-1]   # End of backward trajectory

for p in particles:
    forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
    backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
    
    forward_initial = forward_p[forward_p['Timestep'] == initial_timestep]
    backward_final = backward_p[backward_p['Timestep'] == final_timestep]
  
    delta_initial_final += compute_delta(forward_initial, backward_final)  # Sum over the three particles
print('delta initial final:', delta_initial_final)

# Crossing time 
T_c = mp.mpf(2) * mp.sqrt(2)
# Normalize lifetime 
T_norm = [mp.mpf(timestep) / T_c for timestep in timesteps[:len(delta_per_step)]]

# Phase-space distance over lifetime
plt.figure(figsize=(10, 6))
plt.plot(T_norm, np.flip(delta_per_step), color='b', alpha=0.5)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\delta$')
plt.title('Phase-Space Distance Over Lifetime')
plt.grid(True)
plt.savefig('./figures/phasespacedist.png')

# Cumulative sum of the phase-space distance (delta)
# Flip the order to measure separation from beginning of both trajectories
cumulative_delta = np.cumsum(np.flip(delta_per_step))

# Cumulative distribution of delta over time  
plt.figure(figsize=(10, 6))
plt.plot(T_norm, cumulative_delta, color='g', alpha=0.7, label=r"$\delta$")
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\delta$')
plt.title('Cumulative Distribution of Delta')
plt.grid(True)
plt.yscale('log')
plt.legend()
plt.savefig('./figures/cumulative_delta.png') 
plt.show()

