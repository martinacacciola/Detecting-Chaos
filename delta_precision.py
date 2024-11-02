import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

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

particles = df['Particle Number'].unique()

# Save the original precision to restore later
original_precision = getcontext().prec
getcontext().prec = 100  # Temporarily increase precision for intermediate calculations

# Define phase space distance function with high precision
def compute_delta(forward_state, backward_state):
    # Calculate the phase-space distance with increased precision
    diff_vel = sum((Decimal(vx_f) - Decimal(-vx_b))**2 + (Decimal(vy_f) - Decimal(-vy_b))**2 + (Decimal(vz_f) - Decimal(-vz_b))**2
                   for vx_f, vy_f, vz_f, vx_b, vy_b, vz_b in zip(
                       forward_state['X Velocity'].values, forward_state['Y Velocity'].values, forward_state['Z Velocity'].values,
                       backward_state['X Velocity'].values, backward_state['Y Velocity'].values, backward_state['Z Velocity'].values))

    diff_pos = sum((Decimal(x_f) - Decimal(x_b))**2 + (Decimal(y_f) - Decimal(y_b))**2 + (Decimal(z_f) - Decimal(z_b))**2
                   for x_f, y_f, z_f, x_b, y_b, z_b in zip(
                       forward_state['X Position'].values, forward_state['Y Position'].values, forward_state['Z Position'].values,
                       backward_state['X Position'].values, backward_state['Y Position'].values, backward_state['Z Position'].values))

    # Return summed result, quantized to 90 decimal places at the end
    return (diff_vel + diff_pos).quantize(Decimal('1.' + '0' * 90))

# Compute delta at each symmetric timestep
delta_per_step = []
for i in range(len(timesteps_forward)):
    delta_sum = Decimal(0)
    for p in particles:
        forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
        backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
        
        forward_state = forward_p[forward_p['Timestep'] == timesteps_forward[i]]
        backward_state = backward_p[backward_p['Timestep'] == timesteps_backward[i]]
        
        # Accumulate delta and quantize only after summing
        delta = compute_delta(forward_state, backward_state)
        delta_sum += delta

    delta_per_step.append(delta_sum.quantize(Decimal('1.' + '0' * 90)))  # Quantize only final result

# Restore the original precision
getcontext().prec = original_precision

# We want to compute delta between initial and final states
delta_initial_final = Decimal(0)

initial_timestep = timesteps[0]  # Start of forward trajectory
final_timestep = timesteps[-1]   # End of backward trajectory

for p in particles:
    forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
    backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
    
    forward_initial = forward_p[forward_p['Timestep'] == initial_timestep]
    backward_final = backward_p[backward_p['Timestep'] == final_timestep]
  
    delta_initial_final += compute_delta(forward_initial, backward_final)  # Sum over the three particles

print('Delta initial final:', delta_initial_final)

# Check the number of decimal places in a single Decimal object
# Function to count decimal places for a single Decimal object
def count_decimal_places(num):
    # Convert number to string and split at the decimal point
    str_num = str(num)
    if '.' in str_num:
        # Return count of digits after the decimal point
        return len(str_num.split('.')[1])
    else:
        return 0  # No decimal places if it's an integer

# Check the number of decimal places for each element in delta_per_step
decimal_places_per_step = [count_decimal_places(delta) for delta in delta_per_step]

# Print the entire array of decimal places counts
print("Decimal places for each element in delta_per_step:", decimal_places_per_step)


# Check the number of decimal places in an array of Decimal objects
def count_decimal_places_array(arr):
    decimal_places = []
    for num in arr:
        decimal_places.append(count_decimal_places(num))
    return np.array(decimal_places)

print(count_decimal_places(delta_initial_final))
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
plt.grid(True)
#plt.savefig('./figures/phasespacedist.png')
plt.show()

# Cumulative sum of the phase-space distance (delta)
# Flip the order to measure
cumulative_delta = np.cumsum(np.flip(delta_per_step))
plt.figure(figsize=(10, 6))
plt.plot(T_norm, cumulative_delta, color='g', alpha=0.7, label=r"$\delta$")
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\delta$')
plt.title('Cumulative Distribution of Delta')
plt.grid(True)
plt.yscale('log')
plt.legend()
#plt.savefig('./figures/cumulative_delta.png')
plt.show()


