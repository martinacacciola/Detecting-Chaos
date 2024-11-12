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

# Phase-space distance over lifetime
plt.figure(figsize=(10, 6))
plt.plot(T_norm, np.flip(delta_per_step), color='b', alpha=0.5)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\log10(\delta)$')
plt.yscale('log')
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
plt.ylabel(r'$\log10(\delta)$')
plt.title('Cumulative Distribution of Delta')
plt.grid(True)
plt.yscale('log')
plt.legend()
plt.savefig('./figures/cumulative_delta.png') 
#plt.show()

########
# Amplification factor for each integration step
A = [delta_initial_final/np.flip(delta_per_step[i]) for i in range(len(delta_per_step))]
A_cumul = np.cumsum(A)

# Amplification factor over lifetime (cumulative distribution)
plt.figure(figsize=(10, 6))
plt.plot(T_norm, A_cumul, color='b', alpha=0.5)
plt.xlabel(r'$T/T_c$')
plt.ylabel('A')
plt.yscale('log')
plt.grid(True)
#plt.legend()
plt.title('Amplification Factor (Cumulative Distribution)')
#plt.show()
#plt.savefig('./figures/A.png')


# Metric
# It is the sum of the squared distances between every pair of bodies
# Summed over the three particles

# We want to compute the metric considering squared distances between every pair of bodies (j > i)
# Use i and j as indices to distinguish the particles
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

# Metric evolution over lifetime
plt.figure(figsize=(10, 6))
plt.plot(T_norm, metric_per_step, color='b', alpha=0.5)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\log_{10}(ds^2)$')
plt.yscale('log')
plt.grid(True)
plt.title('Metric Evolution Over Lifetime')
plt.savefig('./figures/metric_evolution.png')
#plt.show()

########
# Instantaneous Lyapunov exponent for each step
lifetime = T_norm[-1] # Total lifetime 
lyapunov_exponents = [mp.log(A[i]) / lifetime for i in range(len(A))]

# Plot the distribution of instantaneous Lyapunov exponents
plt.figure(figsize=(10, 6))
plt.plot(T_norm, lyapunov_exponents, color='r', alpha=0.7)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\lambda$')
plt.title('Instantaneous Lyapunov Exponent Distribution Over Lifetime')
plt.grid(True)
#plt.savefig('./figures/lyapunov_exponent_distribution.png')
plt.show()

lyapunov_exponents_float = [float(le) for le in lyapunov_exponents]
# histogram of the distribution of the instantaneous Lyapunov exponents
plt.figure(figsize=(10, 6))
sns.histplot(lyapunov_exponents_float, bins=40, kde= True, alpha=0.7, edgecolor='black', stat='density')
sns.kdeplot(lyapunov_exponents_float, color='r', linewidth=2)
plt.xlabel('Instantaneous Lyapunov Exponent')
plt.ylabel('Frequency')
plt.title('Distribution of Instantaneous Lyapunov Exponents')
plt.grid(True)
#plt.savefig('./figures/lyapunov_exp.png')
plt.show()



















######## DRAFT #################################################
# plotting the distribution of the finite-time Lyapunov exponents
def plot_lyapunov_timescale_distribution(A, T_total):
    lyapunov_timescales = []

    # Compute finite-time Lyapunov timescale for each timestep
    for amplification_factor in A:
        #if amplification_factor > 0:  # Avoid log(0) cases
        lambda_t = mp.log(amplification_factor) / T_total
        # cumulative distribution for the binning
        lyapunov_timescale = lambda_t #if lambda_t != 0 else mp.inf  # Avoid division by zero # check why it's happening, should be 1/log(lambda)
        lyapunov_timescales.append(float(lyapunov_timescale)) 

    
    # Plotting the distribution of finite-time Lyapunov timescales
    plt.figure(figsize=(10, 6))
    sns.histplot(lyapunov_timescales, bins=30, kde= True, alpha=0.7, edgecolor='black', stat='density')
    sns.kdeplot(lyapunov_timescales, color='r', linewidth=2)
    plt.xlabel('Finite-Time Lyapunov Timescale')
    plt.ylabel('Frequency')
    plt.title('Distribution of Finite-Time Lyapunov Timescales')
    plt.grid(True)
    plt.savefig('./figures/lyapunov_exp.png')
    plt.show()
    
    return lyapunov_timescales


# compute amplification factor for a larger window size (not for each timestep)
window_size = 1000
A_window = [delta_initial_final / np.sum(np.flip(delta_per_step[i:i+window_size])) for i in range(len(delta_per_step) - window_size)]
T_total = T_norm[-1]  # total lifetime
lyapunov_timescales = plot_lyapunov_timescale_distribution(A_window, T_total)



########
# trying with slopes
def compute_lyapunov_timescale(A, T_norm, finite_time_interval):
    """
    Computes the rolling Lyapunov timescale over a finite time interval.

    Parameters:
    A (list of mp.mpf): List of amplification factors.
    T_norm (list of mp.mpf): Normalized time list.
    finite_time_interval (int): Size of the finite time interval.

    Returns:
    list of mp.mpf: List of Lyapunov timescales.
    list of mp.mpf: List of midpoints for plotting.
    """
    lyapunov_timescale = []

    for i in range(len(A) - finite_time_interval + 1):
        A_integral = sum(A[i:i + finite_time_interval])
        T_interval = T_norm[i + finite_time_interval - 1] - T_norm[i]
        
        lyapunov_timescale.append(mp.log(A_integral) / T_interval)
      
    T_midpoints = T_norm[:len(lyapunov_timescale)]  # Match time points for plotting
    return lyapunov_timescale, T_midpoints

def compute_slopes(T_norm, lyapunov_timescale, window_size):
    """
    Computes the slopes of the Lyapunov timescale over specified time windows.

    Parameters:
    T_norm (list of mp.mpf): Normalized time list.
    lyapunov_timescale (list of mp.mpf): List of Lyapunov timescales.
    window_size (int): Size of the time window.

    Returns:
    list of floats: Slopes of the Lyapunov timescale.
    """
    slopes = []
    for i in range(len(T_norm) - window_size):
        delta_lyapunov = lyapunov_timescale[i + window_size] - lyapunov_timescale[i]
        delta_T = T_norm[i + window_size] - T_norm[i]
        slope = float(delta_lyapunov) / float(delta_T) if delta_T != 0 else np.nan
        slopes.append(slope)
    return slopes


def plot_slope_distribution(slopes, window_size):
    """
    Plots the distribution of the slopes of the Lyapunov timescale.

    Parameters:
    slopes (list of floats): Slopes of the Lyapunov timescale.
    window_size (int): Size of the time window.
    """
    # Convert slopes to a numpy array for plotting
    slopes_np = np.array(slopes)

    # Plot the distribution of the slopes with KDE superimposed
    plt.figure(figsize=(10, 6))
    sns.histplot(slopes_np, bins=30, kde=True, alpha=0.7, edgecolor='black')
    #sns.kdeplot(slopes_np, color='red', linewidth=2)
    plt.xlabel('Slope of Lyapunov Timescale')
    plt.ylabel('Density')
    plt.title(f'Distribution of Slopes of Lyapunov Timescale (Window Size = {window_size})')
    plt.grid(True)
    plt.show()



# Assuming `A` and `T_norm` are already defined and computed
finite_time_interval = 10  # Example size, adjust as needed

# Compute Lyapunov timescales
lyapunov_timescale, T_midpoints = compute_lyapunov_timescale(A, T_norm, finite_time_interval)

# Define different time window sizes for slope calculation
window_sizes = [1, 10, 50, 100, 200]

# Compute and plot the distribution of slopes for each window size
#
# for window_size in window_sizes:
    #slopes = compute_slopes(T_midpoints, lyapunov_timescale, window_size)
    #plot_slope_distribution(slopes, window_size)


""" 
## TRYING W SLOPES OF delta

# Using floats
def compute_slopes(T_norm, d, window_size):
    # Convert T_norm and d to standard float arrays
    #T_norm = np.array([float(t) for t in T_norm])
    #d = np.array([float(delta) for delta in d])
    
    slopes = []
    for i in range(0, len(T_norm) - window_size):
        # Calculate the slope over the window
        delta_d = d[i + window_size] - d[i]
        delta_T = T_norm[i + window_size] - T_norm[i]
        slope = delta_d / delta_T
        slopes.append(slope)
    return np.array(slopes)

# Update plot_slope_d to ensure numeric type for plotting
def plot_slope_d(T_norm, d, window_size):
    # Convert T_norm and cumulative_delta to float arrays
    T_norm = np.array([float(t) for t in T_norm])
    d = np.array([float(delta) for delta in d])
    
    slopes_d = compute_slopes(T_norm, d, window_size)
    
    # Plot the distribution of the slopes
    plt.figure(figsize=(10, 6))
    sns.histplot(slopes_d, bins=30, alpha=0.7, label=f'Slope Distribution of delta (window_size={window_size})')
    plt.xlabel(r'Slope of $\delta$ (d$\delta$/dT)')
    plt.ylabel('Frequency')
    plt.title(r'Distribution of Slopes of $\delta$')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_slope_d(T_norm, delta_per_step, 1)
# all values are in zero - so not using this at the moment
# maybe problem in the computation of slope, not precise enough? """
