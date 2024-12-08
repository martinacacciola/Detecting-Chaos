import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# List of trajectory and slope file paths
trajectory_files = [
    './Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv',
    './Brutus data/plummer_triples_L0_00_i1966_e90_Lw392.csv'
    # Add more file paths as needed
]

slope_files = [
    './data/inst_slopes_L0_00_i1775_e90_Lw392.txt',
    './data/inst_slopes_L0_00_i1966_e90_Lw392.txt'
    # Add more file paths as needed
]

# Define the MLP model
mlp_model = Sequential()
mlp_model.add(Input(shape=(18,)))  # 18 features (3 positions + 3 velocities) * 3 particles
mlp_model.add(Dense(64, activation='relu'))
mlp_model.add(Dense(64, activation='relu'))
mlp_model.add(Dense(1, activation='linear'))

# Compile the model
mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_list = []

# Iterate over the files and train the model incrementally
for i, (traj_path, slope_path) in enumerate(zip(trajectory_files, slope_files)):
    # Load the dataset
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
    y = []

    # Exclude the last step of the forward_trajectory
    timesteps = forward_trajectory['Timestep'].unique()[:-1]

    # Align slope_df index with the modified timestep values
    slope_df.index = timesteps

    # Loop through each timestep
    for t in timesteps:
        timestep_data = []
        for p in particles[:3]:  # Take the first three particles
            forward_state = forward_trajectory[
                (forward_trajectory['Particle Number'] == p) & (forward_trajectory['Timestep'] == t)
            ]
            if not forward_state.empty:
                # Combine position and velocity into a single array
                pos_vel = forward_state[pos_vel_cols].values[0]
                timestep_data.extend(pos_vel)  # Add particle's data for this timestep
        
        # Ensure we have data for all 3 particles
        if len(timestep_data) == 18:  # 6 features (3 positions + 3 velocities) per particle * 3 particles
            X.append(timestep_data)
            y.append(slope_df.loc[t, 0])  # Assuming slope_df index matches the timesteps

    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    history = mlp_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)
    history_list.append(history)

    # Plot training & validation loss values for each file
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss for File {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./figures/loss_evolution_file_{i+1}.png')
    plt.show()

# Evaluate the model on the test set
y_pred = mlp_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Test MSE: {mse}')
print(f'Test R^2: {r2}')

# Plot true vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.tight_layout()
plt.savefig('./figures/true_vs_predicted.png')
plt.show()

# Summary plot for average loss across all files
train_losses = np.array([history.history['loss'] for history in history_list])
val_losses = np.array([history.history['val_loss'] for history in history_list])

mean_train_loss = np.mean(train_losses, axis=0)
std_train_loss = np.std(train_losses, axis=0)
mean_val_loss = np.mean(val_losses, axis=0)
std_val_loss = np.std(val_losses, axis=0)

epochs = range(1, len(mean_train_loss) + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs, mean_train_loss, label='Mean Training Loss')
plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.2)
plt.plot(epochs, mean_val_loss, label='Mean Validation Loss')
plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.2)
plt.title('Average Model Loss Across All Files')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('./figures/average_loss.png')
plt.show()