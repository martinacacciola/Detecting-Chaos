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
#print(len(timesteps_forward), len(timesteps_backward))

particles = df['Particle Number'].unique()

# Phase space distance between two states
def compute_delta(forward_state, backward_state):
    # Extract positions
    x_f, y_f, z_f = forward_state['X Position'].values, forward_state['Y Position'].values, forward_state['Z Position'].values
    x_b, y_b, z_b = backward_state['X Position'].values, backward_state['Y Position'].values, backward_state['Z Position'].values

    # Extract velocities
    vx_f, vy_f, vz_f = forward_state['X Velocity'].values, forward_state['Y Velocity'].values, forward_state['Z Velocity'].values
    # Flip the sign of the velocities for the backward state
    vx_b, vy_b, vz_b = -backward_state['X Velocity'].values, -backward_state['Y Velocity'].values, -backward_state['Z Velocity'].values

    # Compute the phase-space distance for this particle
    diff_vel = (vx_f - vx_b)**2 + (vy_f - vy_b)**2 + (vz_f - vz_b)**2
    diff_pos = (x_f - x_b)**2 + (y_f - y_b)**2 + (z_f - z_b)**2
    
    return diff_vel + diff_pos


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
        delta_sum += delta
        
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
def count_decimal_places(arr):
    decimal_places = []
    for num in arr.flatten():
        # Convert number to string
        str_num = str(num)
        if '.' in str_num:
            # Count characters after the decimal point
            decimal_places.append(len(str_num.split('.')[1]))
        else:
            decimal_places.append(0)  # No decimal places for integers
    return np.array(decimal_places)

print(count_decimal_places(delta_initial_final))
print(count_decimal_places(np.array(delta_per_step)))


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
#plt.semilogy()
plt.legend()
plt.savefig('./figures/cumulative_delta.png') 
plt.show()

# Amplification factor for each integration step
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





