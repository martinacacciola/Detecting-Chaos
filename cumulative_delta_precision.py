import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from scipy.stats import ks_2samp

# Set global precision for Decimal
getcontext().prec = 90

# Load data and convert specified columns to Decimal
df = pd.read_csv('./Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv')  # Load as strings first

# Convert columns to Decimal
for col in ['Timestep', 'Mass', 'X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']:
    df[col] = df[col].apply(lambda x: Decimal(x))


forward_trajectory = df[df['Phase'] == 1]
backward_trajectory = df[df['Phase'] == -1]

total_timesteps = len(df['Timestep'].unique())
midpoint = total_timesteps // 2  # Midpoint corresponds to t=T=lifetime
timesteps = df['Timestep'].unique()

# Define symmetric timesteps
timesteps_forward = timesteps[:midpoint]
timesteps_backward = timesteps[midpoint+1:][::-1]  # Reverse order for symmetry
#print(len(timesteps_forward), len(timesteps_backward))

particles = df['Particle Number'].unique()

# Phase space distance between two states
def compute_delta(forward_state, backward_state):
    # Convert each element to Decimal and calculate the phase-space distance
    # Flip the sign of the velocities for the backward state
    diff_vel = sum((Decimal(vx_f) - Decimal(-vx_b))**2 + (Decimal(vy_f) - Decimal(-vy_b))**2 + (Decimal(vz_f) - Decimal(-vz_b))**2
                   for vx_f, vy_f, vz_f, vx_b, vy_b, vz_b in zip(
                       forward_state['X Velocity'].values, forward_state['Y Velocity'].values, forward_state['Z Velocity'].values,
                       backward_state['X Velocity'].values, backward_state['Y Velocity'].values, backward_state['Z Velocity'].values))

    diff_pos = sum((Decimal(x_f) - Decimal(x_b))**2 + (Decimal(y_f) - Decimal(y_b))**2 + (Decimal(z_f) - Decimal(z_b))**2
                   for x_f, y_f, z_f, x_b, y_b, z_b in zip(
                       forward_state['X Position'].values, forward_state['Y Position'].values, forward_state['Z Position'].values,
                       backward_state['X Position'].values, backward_state['Y Position'].values, backward_state['Z Position'].values))
     # Compute the phase-space distance for this particle
    return Decimal(diff_vel) + Decimal(diff_pos)


# Flip the order of the backward trajectory
#backward_trajectory = backward_trajectory.iloc[::-1]
#print('Backward traj head:',backward_trajectory.head())

# Compute delta at each symmetric timestep
delta_per_step = []

for i in range(len(timesteps_forward)):
    delta_sum = 0  
    for p in particles:
        forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
        backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
        
        forward_state = forward_p[forward_p['Timestep'] == timesteps_forward[i]]
        backward_state = backward_p[backward_p['Timestep'] == timesteps_backward[i]]
        
        # check the comparison - seems correct
        if i == 30 and p == particles[0]:  # check for a timestep and for a particle
            print("Forward state compared:", forward_state)
            print("Backward state compared:", backward_state)
    
        delta = compute_delta(forward_state, backward_state)
        delta_sum += Decimal(delta)
        
    delta_per_step.append(delta_sum) # Append the delta summed over the bodies for this timestep
    #print(delta_per_step)

# We want to compute delta between initial and final states
delta_initial_final = 0

initial_timestep = timesteps[0]  # start of forward trajectory
final_timestep = timesteps[-1]   # end of backward trajectory

for p in particles:
    forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
    backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
    
    forward_initial = forward_p[forward_p['Timestep'] == initial_timestep]
    backward_final = backward_p[backward_p['Timestep'] == final_timestep]
  
    delta_initial_final += compute_delta(forward_initial, backward_final) # Sum over the three particles
print('delta initial final:', delta_initial_final)
# check the number of decimal places in delta initial final (array)
def count_decimal_places_single(num):
    # Convert number to string
    str_num = str(num)
    if '.' in str_num:
        # Count characters after the decimal point
        return len(str_num.split('.')[1])
    else:
        return 0  # No decimal places for integers

# Check the number of decimal places in an array of Decimal objects
def count_decimal_places_array(arr):
    decimal_places = []
    for num in arr:
        decimal_places.append(count_decimal_places_single(num))
    return np.array(decimal_places)

print(count_decimal_places_single(delta_initial_final))
print(count_decimal_places_array(delta_per_step))


# Crossing time as Decimal for high precision
T_c = Decimal(2) * Decimal(np.sqrt(2))

# Normalize lifetime using Decimal division
T_norm = [Decimal(timesteps[i]) / T_c for i in range(len(delta_per_step))]

# Phase-space distance over lifetime
plt.figure(figsize=(10, 6))
plt.plot(T_norm, np.flip(delta_per_step), color='b', alpha=0.5)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\delta$')
#plt.yscale('log')
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
# to see after the 0-behaviour, to last time value
#plt.xlim(784, T_norm[-1])
#plt.semilogy()
plt.legend()
plt.savefig('./figures/cumulative_delta.png') 
plt.show()


############################################################################################################
# working only on the final part of the curve (skipping all the zeros)
delta_flip = np.flip(delta_per_step)
d = delta_flip[784:]
cumulative_delta = np.cumsum(d)
T_norm = T_norm[784:]


# Now we want to understand if the distribution of slopes is the same on =/ parts of the curve
# We divide the curve in more time windows and compare the distributions of the slopes
def compare_distribution(cumulative_delta, T_norm, num_windows):
    # Split data into sections
    window_size = len(T_norm) // num_windows

    T_norm_windows = [T_norm[i * window_size:(i + 1) * window_size] for i in range(num_windows)]
    cumulative_delta_windows = [cumulative_delta[i * window_size:(i + 1) * window_size] for i in range(num_windows)]
    
    # Adjust each window to start from T=0
    T_norm_windows_adjusted = [[t - T_window[0] for t in T_window] for T_window in T_norm_windows]

    # Plot each window
    plt.figure(figsize=(10, 6))
    colors = ['r', 'b', 'g', 'm', 'c', 'y']
    for i in range(num_windows):
        plt.plot(T_norm_windows_adjusted[i], cumulative_delta_windows[i], color=colors[i % len(colors)], label=f'Window {i + 1}', alpha=0.7)
    
    plt.xlabel(r'$T/T_c$')
    plt.ylabel(r'$\log_{10}(\delta)$')
    plt.title(f'Comparison of {num_windows} Sections of Cumulative Delta')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Perform the KS test between each pair of windows
    print(f"KS Test for {num_windows} sections of the curve:")
    for i in range(num_windows):
        for j in range(i + 1, num_windows):
            ks_stat, p_value = ks_2samp(cumulative_delta_windows[i], cumulative_delta_windows[j])
            print(f"KS Statistic (Window {i + 1} vs Window {j + 1}): {ks_stat}, P-value: {p_value}")

# Call the function with cumulative_delta
compare_distribution(cumulative_delta, T_norm, 4)


















# Amplification factor for each integration step
# the problem now is the division by zero - so not using this at the moment
#A = [delta_initial_final/np.flip(delta_per_step[i]) for i in range(len(delta_per_step))]

# Amplification factor over lifetime
#plt.figure(figsize=(10, 6))
#plt.plot(T_norm, A, color='b', alpha=0.5)
#plt.xlabel(r'$T/T_c$')
#plt.ylabel(r'$\delta$')
#plt.xlim(0, 100)
#plt.yscale('log')
#plt.ylim(0, 60)
#plt.grid(True)
#plt.title('Amplification Factor')
#plt.show()
#plt.savefig('./figures/A.png')

#cumulative_A = np.cumsum(A)
# Cumulative distribution of the amplification factor over time  
##plt.figure(figsize=(10, 6))
#plt.plot(T_norm, cumulative_A, color='g', alpha=0.7, label='A')
#plt.xlabel(r'$T/T_c$')
#plt.ylabel(r'$\delta$')
#plt.title('Cumulative Distribution of A')
#plt.grid(True)
#plt.yscale('log')
#plt.semilogy()
#plt.legend()
#plt.savefig('./figures/cumulative_A.png') 
#plt.show() 





