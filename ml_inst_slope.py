import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

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

# Identify unique particles
particles = forward_trajectory['Particle Number'].unique()

# Initialize X with 18 features for all particles combined
X = []

for t in forward_trajectory['Timestep'].unique()[:-1]:  # Exclude the last timestep since y is one less
    timestep_data = []
    for p in particles:
        forward_state = forward_trajectory[(forward_trajectory['Particle Number'] == p) & (forward_trajectory['Timestep'] == t)]
        
        # Combine position and velocity into a single array
        pos_vel = forward_state[pos_vel_cols].values[0]
        timestep_data.extend(pos_vel)  # Add to timestep's data
    
    X.append(timestep_data)

# Convert to numpy array
X = np.array(X)

# Use the instantaneous slope as y
instantaneous_slope = slope_df.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, instantaneous_slope, test_size=0.2, random_state=42)

# Define the MLP model
mlp_model = Sequential()
mlp_model.add(Input(shape=(X_train.shape[1],)))
mlp_model.add(Dense(64, activation='relu'))
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
