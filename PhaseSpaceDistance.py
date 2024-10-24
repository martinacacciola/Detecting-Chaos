import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp

df = pd.read_csv('./Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv')

forward_trajectory = df[df['Phase'] == 1]
backward_trajectory = df[df['Phase'] == -1]

total_timesteps = len(df['Timestep'].unique())
midpoint = total_timesteps // 2  # Midpoint corresponds to t=T=lifetime
timesteps = df['Timestep'].unique()

particles = df['Particle Number'].unique()

# Phase space distance between two states
def compute_delta(forward_state, backward_state):
    # Extract positions
    x_f, y_f, z_f = forward_state['X Position'].values, forward_state['Y Position'].values, forward_state['Z Position'].values
    x_b, y_b, z_b = backward_state['X Position'].values, backward_state['Y Position'].values, backward_state['Z Position'].values

    # Extract velocities
    vx_f, vy_f, vz_f = forward_state['X Velocity'].values, forward_state['Y Velocity'].values, forward_state['Z Velocity'].values
    vx_b, vy_b, vz_b = backward_state['X Velocity'].values, backward_state['Y Velocity'].values, backward_state['Z Velocity'].values

    # Compute the phase-space distance for this particle
    diff_vel = (vx_f - vx_b)**2 + (vy_f - vy_b)**2 + (vz_f - vz_b)**2
    diff_pos = (x_f - x_b)**2 + (y_f - y_b)**2 + (z_f - z_b)**2
    
    return np.sum(diff_vel + diff_pos)

# We want to compute delta btw backward and forward solutions at each integration step
# We need to consider the states symetrically around the midpoint
# backward state = total timesteps - 1 - current forward state
# forward state = current forward state (until midpoint)
delta_per_step = []

for i in range(len(timesteps) // 2):
    delta_sum = 0  
    for p in particles:
        forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
        backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
        
        forward_state = forward_p[forward_p['Timestep'] == timesteps[i]]
        backward_state = backward_p[backward_p['Timestep'] == timesteps[total_timesteps - 1 - i]]
        
        delta = compute_delta(forward_state, backward_state)
        delta_sum += delta
        
    delta_per_step.append(delta_sum)  # Append the delta summed over the bodies for this timestep

# We want to compute delta between initial and final states
delta_initial_final = 0

for p in particles:
    forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
    backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
    
    # Initial state from forward trajectory
    forward_initial = forward_p[forward_p['Timestep'] == 0]
    
    # Final state from backward trajectory
    max_backward_timestep = backward_p['Timestep'].max()
    backward_final = backward_p[backward_p['Timestep'] == max_backward_timestep]
    
    delta_initial_final += compute_delta(forward_initial, backward_final)

# Histogram of the distribution of delta values
plt.figure(figsize=(10, 6))
plt.hist(delta_per_step, bins=100, color='b', alpha=0.7, edgecolor='black')
plt.title('Distribution of Phase-Space Distance')
plt.xlabel(r'$\Delta^2$')
plt.ylabel('Frequency')
plt.xlim(0,100)
plt.grid(True)
plt.savefig('./figures/delta_histogram.png') # Not sure this plot is meaningful

# Compute and sort in ascending order
log_delta = np.sort(np.log10(delta_per_step))
# Compute the CDF: cumulative probability for each value
cdf = np.arange(1, len(log_delta) + 1) / len(log_delta)

plt.figure(figsize=(10, 6))
plt.plot(log_delta, cdf, color='g', alpha=0.7, label=r"CDF of $\log_{10}(\Delta^2)$")
plt.xlabel(r'$\log_{10}(\Delta^2)$')
plt.ylabel(r'$f_{CDF}$')
plt.title('Cumulative Distribution Function of Amplification Factor')
plt.grid(True)
plt.xlim(min(log_delta), max(log_delta))
plt.ylim(0, 1)
plt.legend()
plt.savefig('./figures/cdf_delta.png')

# Amplification factor for each integration step
A = [delta_initial_final/delta_per_step[i] for i in range(len(delta_per_step))]

# Crossing time 
T_c = 2 * np.sqrt(2)
# Normalize lifetime 
T_norm = [timesteps[i] / T_c for i in range(len(delta_per_step))]

# Amplification factor evolution over lifetime
plt.figure(figsize=(10, 6))
plt.plot(T_norm, np.log10(A), color='b', alpha=0.5)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\log_{10}(A)$')
plt.xlim(0, 100)
#plt.ylim(0, 60)
plt.grid(True)
plt.savefig('./figures/amplification_factor.png')

# Cumulative sum of the amplification factor (A)
cumulative_A = np.cumsum(A)

# Cumulative distribution of the amplification factor over time  
plt.figure(figsize=(10, 6))
plt.plot(T_norm, np.log10(cumulative_A), color='g', alpha=0.7, label="A")
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\log_{10}(A)$')
plt.title('Cumulative Distribution of Amplification Factor Over Time')
plt.grid(True)
plt.legend()
plt.savefig('./figures/cumulative_A.png') 

# Now we want to understand if the distribution of slopes is the same on =/ parts of the curve
# We divide the curve in more time windows and compare the distributions of the slopes

def compare_distribution(num_windows):
    # Split data into sections
    window_size = len(T_norm) // num_windows

    T_norm_windows = [T_norm[i * window_size:(i + 1) * window_size] for i in range(num_windows)]
    A_windows = [A[i * window_size:(i + 1) * window_size] for i in range(num_windows)]
    
    # Adjust each window to start from T=0
    T_norm_windows_adjusted = [[t - T_norm_window[0] for t in T_norm_window] for T_norm_window in T_norm_windows]

    # Compute cumulative sums for each window
    cumulative_A_windows = [np.cumsum(A_window) for A_window in A_windows]
    
    plt.figure(figsize=(10, 6))
    colors = ['r', 'b', 'g', 'm', 'c', 'y']  # Add more colors if num_windows > 6
    for i in range(num_windows):
        plt.plot(T_norm_windows_adjusted[i], np.log10(cumulative_A_windows[i]), color=colors[i % len(colors)], label=f'Window {i + 1}', alpha=0.7)
    
    plt.xlabel(r'$T/T_c$')
    plt.ylabel(r'$\log_{10}(A)$')
    plt.title(f'Comparison of {num_windows} Sections of Cumulative Amplification Factor')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./figures/compare_{num_windows}_sections_cumulative_A.png')
    
    # Perform the KS test between each pair of windows
    print(f"KS Test for {num_windows} sections of the curve:")
    for i in range(num_windows):
        for j in range(i + 1, num_windows):
            ks_stat, p_value = ks_2samp(A_windows[i], A_windows[j])
            print(f"KS Statistic (Window {i + 1} vs Window {j + 1}): {ks_stat}, P-value: {p_value}")

compare_distribution(4) 


#######

# Compute and sort in ascending order
log_A = np.sort(np.log10(A))
# Compute the CDF: cumulative probability for each value
cdf = np.arange(1, len(log_A) + 1) / len(log_A)

plt.figure(figsize=(10, 6))
plt.plot(log_A, cdf, color='g', alpha=0.7, label="CDF of log10(A)")
plt.xlabel(r'$\log_{10}(A)$')
plt.ylabel(r'$f_{CDF}$')
plt.title('Cumulative Distribution Function of Amplification Factor')
plt.grid(True)
plt.xlim(min(log_A), max(log_A))
plt.ylim(0, 1)
plt.legend()
plt.savefig('./figures/cdf_A.png')

# Metric
# It is the sum of the squared distances between every pair of bodies
# Summed over the three particles

# We want to compute the metric considering squared distances between every pair of bodies (j > i)
# Use i and j as indices to distinguish the particles
# Trajectory to decide on backward or forward ??
def compute_metric(trajectory, timestep):
    ds = 0
    particles_positions = []
    
    for p in particles:
        particle_state = trajectory[(trajectory['Particle Number'] == p) & (trajectory['Timestep'] == timestep)]
        x, y, z = particle_state['X Position'].values, particle_state['Y Position'].values, particle_state['Z Position'].values
        particles_positions.append((x, y, z))
    
    for i in range(len(particles_positions)):
        for j in range(i+1, len(particles_positions)):
            xi, yi, zi = particles_positions[i]
            xj, yj, zj = particles_positions[j]
            ds += (xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2
           
    return ds

# We want to compute the metric for each integration step
# Using the forward trajectory 
metric_per_step = []
for i in range(len(timesteps) // 2):
    metric = compute_metric(forward_trajectory, timesteps[i])
    metric_per_step.append(metric)

T_norm_metric = [timesteps[i] / T_c for i in range(len(metric_per_step))]
# Metric evolution over lifetime
plt.figure(figsize=(10, 6))
plt.plot(T_norm_metric, np.log10(metric_per_step), color='b', alpha=0.5)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\log_{10}(ds^2)$')
plt.grid(True)
plt.savefig('./figures/metric_evolution.png')

T_norm = np.array([timesteps[i] / T_c for i in range(len(delta_per_step))])
# Lyapunov exponent, for each integration step
l_exponent = np.array([np.log10(A[i]) / timesteps[i] for i in range(len(A))])
# Lyapunov timescale (the inverse)
l_timescale = np.array([1 / l_exponent[i] for i in range(len(l_exponent))])

# CDF of the Lyapunov timescale (log T_lambda/T_c)
log_l_timescale = np.sort(np.log10(l_timescale/T_norm))
cdf_l_timescale = np.arange(1, len(log_l_timescale) + 1) / len(log_l_timescale)

plt.figure(figsize=(10, 6))
plt.plot(log_l_timescale, cdf_l_timescale, color='g', alpha=0.7, label=r"CDF of $\log10(T_\lambda/T_c)$")
plt.xlabel(r'$\log10(T_\lambda/T_c)$')
plt.ylabel(r'$f_{CDF}$')
plt.title('Cumulative Distribution Function of Lyapunov Timescale')
plt.grid(True)
plt.xlim(min(log_l_timescale), max(log_l_timescale))
plt.ylim(0, 1)
plt.legend()
plt.savefig('./figures/cdf_t_lambda.png')







