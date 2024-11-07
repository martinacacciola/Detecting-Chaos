import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from mpmath import mp

pd.options.display.float_format = "{:,.120f}".format 

df = pd.read_csv('./Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv', dtype=str) #float_precision='round_trip'
print('the first 5 positions are:', df['X Position'].head())
print('the first 5 velocities are:', df['X Velocity'].head())
print('the decimal places in the first 5 positions are:', df['X Position'].apply(lambda x: len(x.split('.')[1])).head())

def count_decimals(value):
    if '.' in value:
        return len(value.split('.')[1])
    return 0
decimal_places = df['X Position'].apply(count_decimals).head()
print('the decimal places in the first 5 positions are:', decimal_places)
#until this point right number of decimals

# Find the maximum number of decimal places in the necessary columns
max_decimal_places = 0
for col in ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']:
    max_decimal_places = max(max_decimal_places, df[col].apply(count_decimals).max())

# Set the global decimal places
mp.dps = max_decimal_places
mp.prec = 3.33 * max_decimal_places  # Set precision to 3.33 times the decimal places
print(f"Global decimal places set to: {mp.dps}")

def string_to_mpf(value):
    return mp.mpf(value)

for col in ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']:
    df[col] = df[col].apply(string_to_mpf)

# check the precision of the first 5 positions
# Check the precision of the first 5 positions after conversion
def count_decimal_places_mpf(value):
    value_str = mp.nstr(value, max_decimal_places)
    if '.' in value_str:
        return len(value_str.split('.')[1])
    return 0

decimal_places_after = df['X Position'].apply(count_decimal_places_mpf).head()
print('The decimal places in the first 5 positions after conversion are:', decimal_places_after)

# Verify the values with high precision
#for col in ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']:
    #for index, value in enumerate(df[col].head()):
        #print(f'Original {col} [{index}]: {mp.nstr(value, max_decimal_places)}')


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
        # if i == 30 and p == particles[0]:  # Check for a timestep and for a particle
            # print("Forward state compared:", forward_state)
            # print("Backward state compared:", backward_state)
    
        delta = compute_delta(forward_state, backward_state)
        delta_sum += delta
        
    delta_per_step.append(delta_sum)  # Append the delta summed over the bodies for this timestep

# Check precision for the first 10 results of delta per step
    if i < 10:
        print(f"Precision of delta_sum at timestep {i}: {mp.nstr(delta_sum, max_decimal_places)}")

# Print the precision of the first 10 delta_per_step results
for j in range(min(10, len(delta_per_step))):
    print(f"delta_per_step[{j}]: {mp.nstr(delta_per_step[j], max_decimal_places)}")

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


