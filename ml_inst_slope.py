import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
traj_path = './Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv'
slope_path = './data/inst_slopes_L0_00_i1775_e90_Lw392.txt'
traj_df = pd.read_csv(traj_path)
slope_df = pd.read_csv(slope_path, header=None)

# Identify position and velocity columns
pos_vel_cols = ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']

# Separate forward and backward trajectories
forward_trajectory = traj_df[traj_df['Phase'].astype(int) == 1]
backward_trajectory = traj_df[traj_df['Phase'].astype(int) == -1]

particles = forward_trajectory['Particle Number'].unique()

# Group positions and velocities by particle for the forward trajectory
# Assuming timesteps_forward and timesteps_backward are defined and contain the timesteps of interest
timesteps_forward = forward_trajectory['Timestep'].unique()
timesteps_backward = backward_trajectory['Timestep'].unique()

# Initialize dictionaries to store positions and velocities for each particle
forward_positions = {}
forward_velocities = {}
backward_positions = {}
backward_velocities = {}

for p in particles:
    forward_p = forward_trajectory[forward_trajectory['Particle Number'] == p]
    backward_p = backward_trajectory[backward_trajectory['Particle Number'] == p]
    
    # Initialize lists to store positions and velocities for all timesteps
    forward_positions[p] = []
    forward_velocities[p] = []
    backward_positions[p] = []
    backward_velocities[p] = []
    
    for t in timesteps_forward:
        forward_state = forward_p[forward_p['Timestep'] == t]
        if not forward_state.empty:
            forward_positions[p].append(forward_state[['X Position', 'Y Position', 'Z Position']].values[0])
            forward_velocities[p].append(forward_state[['X Velocity', 'Y Velocity', 'Z Velocity']].values[0])
    
    for t in timesteps_backward:
        backward_state = backward_p[backward_p['Timestep'] == t]
        if not backward_state.empty:
            backward_positions[p].append(backward_state[['X Position', 'Y Position', 'Z Position']].values[0])
            backward_velocities[p].append(backward_state[['X Velocity', 'Y Velocity', 'Z Velocity']].values[0])

# Convert lists to numpy arrays
for p in particles:
    forward_positions[p] = np.array(forward_positions[p])
    forward_velocities[p] = np.array(forward_velocities[p])
    backward_positions[p] = np.array(backward_positions[p])
    backward_velocities[p] = np.array(backward_velocities[p])

# Example: Access positions and velocities for a specific particle
selected_particle = particles[0]  # Example: select the first particle
positions_selected = forward_positions[selected_particle]
velocities_selected = forward_velocities[selected_particle]

# Combine positions and velocities into a single array for the selected particle
X = np.hstack((positions_selected, velocities_selected))

# Calculate the instantaneous slope (e.g., using the magnitude of the velocity vector)
instantaneous_slope = np.linalg.norm(velocities_selected, axis=1)

# Set y as the instantaneous slope
y = instantaneous_slope

# Print shapes to verify
print('X shape:', X.shape)
print('y shape:', y.shape)

# X_train contains 18 features and y_train contains the instantaneous slope of phase space distance

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MLP model
mlp_model = Sequential()
mlp_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
mlp_model.add(Dense(64, activation='relu'))
mlp_model.add(Dense(1, activation='linear'))

# Compile the model
mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = mlp_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

# Evaluate the model on the test set
y_pred = mlp_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Test MSE: {mse}')
print(f'Test R^2: {r2}')

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot true vs predicted values
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(True)

plt.tight_layout()
plt.show()

