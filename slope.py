import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from mpmath import mp

pd.options.display.float_format = "{:,.120f}".format 

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

# Get unique particles
particles = df['Particle Number'].unique()

# Phase space distance between two states
def compute_delta(forward_state, backward_state):
    # Extract positions and convert to mpmath
    x_f, y_f, z_f = forward_state['X Position'].values, forward_state['Y Position'].values, forward_state['Z Position'].values
    x_b, y_b, z_b = backward_state['X Position'].values, backward_state['Y Position'].values, backward_state['Z Position'].values

    # Extract velocities and convert to mpmath
    vx_f, vy_f, vz_f = forward_state['X Velocity'].values, forward_state['Y Velocity'].values, forward_state['Z Velocity'].values
    vx_b, vy_b, vz_b = -backward_state['X Velocity'].values, -backward_state['Y Velocity'].values, -backward_state['Z Velocity'].values

    # Compute the phase-space distance for this particle using mpmath
    diff_vel = [mp.mpf((vx_f[i] - vx_b[i])**2 + (vy_f[i] - vy_b[i])**2 + (vz_f[i] - vz_b[i])**2) for i in range(len(vx_f))]
    diff_pos = [mp.mpf((x_f[i] - x_b[i])**2 + (y_f[i] - y_b[i])**2 + (z_f[i] - z_b[i])**2) for i in range(len(x_f))]

    delta = mp.sqrt(sum(diff_vel) + sum(diff_pos))
    
    return delta

# Compute delta at each symmetric timestep
delta_per_step = []

for i in range(len(timesteps_forward)):
    delta_sum = mp.mpf(0)  # Initialize as mp.mpf for high precision
    for p in particles:
        forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
        backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
        
        forward_state = forward_p[forward_p['Timestep'] == timesteps_forward[i]]
        backward_state = backward_p[backward_p['Timestep'] == timesteps_backward[i]]
    
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

 
# Crossing time 
T_c = mp.mpf(2) * mp.sqrt(2)
# Normalize lifetime 
T_norm = [mp.mpf(timestep) / T_c for timestep in timesteps[:len(delta_per_step)]]

# Compute instantaneous slope of phase-space distance
delta_flip = np.flip(delta_per_step)
delta_log = np.log10(np.array(delta_flip, dtype=float))
#delta_arr = np.array(delta_per_step, dtype=float)
slopes = np.diff(delta_log) / np.diff(T_norm)  # Slope calculation using consecutive differences
slopes = np.array(slopes, dtype=float)

# midpoints of T_norm intervals for plotting
T_norm_midpoints = [(T_norm[i] + T_norm[i + 1]) / 2 for i in range(len(T_norm) - 1)]

# distribution of slopes over time  
plt.figure(figsize=(10, 6))
plt.plot(T_norm_midpoints, slopes, color='g', alpha=0.7)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\log10(slope))$')
plt.title('Distribution of slopes over time')
plt.grid(True)
plt.legend()
plt.savefig('./figures/slopes_over_time.png') 
#plt.show()

# Plot histogram of slopes
plt.figure(figsize=(8, 6))
plt.hist(slopes, bins=100, alpha=0.7)
plt.xlabel('Instantaneous Slope of log(delta)')
plt.ylabel('Frequency')
plt.title('Distribution of Instantaneous Slope of Phase-Space Distance')
plt.grid(True)
plt.savefig('./figures/slope_manually.png')
plt.show()


# Compute the indices for the windows
#window_indices = range(0, len(T_norm), window_size)

# Define window size in terms of timesteps (number of points to include in each fit)
window_size = 4

window_slopes = []
window_midpoints = []

# iterate over the indices of T_norm in steps of window_size
for start_idx in range(0, len(T_norm) - window_size + 1, window_size):
    end_idx = start_idx + window_size

    delta_flip = np.flip(delta_per_step)
    # select the window over which computing the slopef
    delta_window = delta_flip[start_idx:end_idx]
    T_norm_window = T_norm[start_idx:end_idx]

    delta_log_window = np.log10(np.array(delta_window, dtype=float))
    
    # compute slope using initial and final values of each window
    slope = (delta_log_window[-1] - delta_log_window[0]) / (T_norm_window[-1] - T_norm_window[0])
    window_slopes.append(float(slope))  

    # midpoint of the current time window
    window_midpoints.append((T_norm_window[0] + T_norm_window[-1]) / 2)

plt.figure(figsize=(10, 6))
plt.plot(window_midpoints, window_slopes, color='b', alpha=0.7)
plt.xlabel(r'$T/T_c$')
plt.ylabel('Slope')
plt.title('Slope of log(delta) Over Time')
plt.grid(True)
plt.savefig('./figures/slope_window.png')
plt.show()

# Plot histogram of slopes
plt.figure(figsize=(8, 6))
plt.hist(window_slopes, bins=50, alpha=0.7)
plt.xlabel('Instantaneous Slope of log(delta)')
plt.ylabel('Frequency')
plt.title('Distribution of Slope of Phase-Space Distance')
plt.grid(True)
plt.savefig('./figures/slope_window_hist.png')
plt.show()

""" # Convert T_norm to float
T_norm_float = [float(t) for t in T_norm]

# Define window size (number of points to include in each fit)
window_size = 300  # Adjust this value as needed

# Initialize list to store slopes
slopes_window = []

# Compute slope for each timestep using a sliding window
for i in range(len(T_norm_float) - window_size + 1):
    # Define the window
    window_T = T_norm_float[i:i + window_size]
    window_delta_log = delta_log[i:i + window_size]
    
    # Fit a linear polynomial (degree 1) to the data in the window
    coefficients = np.polyfit(window_T, window_delta_log, 1)
    
    # Extract the slope (first coefficient)
    slope = coefficients[0]
    
    # Store the slope
    slopes_window.append(slope)

# Compute midpoints of T_norm intervals for plotting
T_norm_midpoints = [(T_norm_float[i] + T_norm_float[i + window_size - 1]) / 2 for i in range(len(T_norm_float) - window_size + 1)]

# Plot the slopes against the midpoints
plt.figure(figsize=(10, 6))
plt.plot(T_norm_midpoints, slopes_window)
plt.xlabel(r'$T/T_c$')
plt.ylabel('Slope')
plt.title('Slope of log(delta) Over Time')
plt.savefig('./figures/slope_window.png')
plt.grid(True)
plt.show()

# Plot histogram of slopes
plt.figure(figsize=(8, 6))
plt.hist(slopes_window, bins=30, alpha=0.7)
plt.xlabel('Instantaneous Slope of log(delta)')
plt.ylabel('Frequency')
plt.title('Distribution of Slope of Phase-Space Distance')
plt.savefig('./figures/slope_window_hist.png')
plt.grid(True)
plt.show() """

