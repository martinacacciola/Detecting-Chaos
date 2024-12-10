import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import seaborn as sns

# TODO: 
# 1) instead of mapping to instantaneous slopes do it with parameters
# to each trajectory map the unique parameters
# 2) change the input s.t. 1 particle is always at the origin, the 2 on x axis and the 3 rotated accordingly

# the network is learning from one coordinate at a time

def process_dataset(traj_path, slope_path, pos_vel_cols, particles):
    # Load the dataset
    traj_df = pd.read_csv(traj_path)
    slope_df = pd.read_csv(slope_path, header=None)

    # Separate forward and backward trajectories
    forward_trajectory = traj_df[traj_df['Phase'].astype(int) == 1]
    backward_trajectory = traj_df[traj_df['Phase'].astype(int) == -1]

    # Exclude the last step of the forward_trajectory bc slope_df has one less timestep
    timesteps = forward_trajectory['Timestep'].unique()[:-1]

    # Align slope_df index with the modified timestep values
    slope_df.index = timesteps

    # Initialize X with 18 features for all particles combined
    X = []
    y = []

    # Loop through each timestep
    for t in timesteps:
        timestep_data = []
        for p in particles:  
            forward_state = forward_trajectory[(forward_trajectory['Particle Number'] == p) & (forward_trajectory['Timestep'] == t)]
            
            # Combine position and velocity into a single array
            pos_vel = forward_state[pos_vel_cols].values[0]
            timestep_data.extend(pos_vel)  # Add particle's data for this timestep
        
       
        X.append(timestep_data)
        y.append(slope_df.loc[t, 0])  # to match slopes index with timestep

    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

# Define the MLP model
# linear stack of layers
mlp_model = Sequential()

# Input layer
mlp_model.add(Input(shape=(18,)))  # 18 features

# First hidden layer
mlp_model.add(Dense(256, activation='relu'))  # of neurons
mlp_model.add(BatchNormalization())        
mlp_model.add(Dropout(0.3))  # rate of input units to drop during training              

# Second hidden layer
mlp_model.add(Dense(256, activation='relu'))  
mlp_model.add(BatchNormalization())         
mlp_model.add(Dropout(0.3))                 

# Third hidden layer
mlp_model.add(Dense(128, activation='relu'))
mlp_model.add(BatchNormalization())
mlp_model.add(Dropout(0.2))

# Fourth hidden layer
mlp_model.add(Dense(128, activation='relu'))
mlp_model.add(BatchNormalization())
mlp_model.add(Dropout(0.2))

# Fifth hidden layer
mlp_model.add(Dense(64, activation='relu'))
mlp_model.add(BatchNormalization())
mlp_model.add(Dropout(0.2))

# Output layer
mlp_model.add(Dense(1, activation='linear'))  # Regression output

# Compile the model
mlp_model.compile(
    optimizer=Adam(learning_rate=0.001),  
    loss='mse',
    metrics=['mae']
)

history_list = []

# Specify the files to be used for training and validation
train_trajectory_files = [
    './Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv',
    './Brutus data/plummer_triples_L0_00_i1966_e90_Lw392.csv'
]

train_slope_files = [
    './data/inst_slopes_L0_00_i1775_e90_Lw392.txt',
    './data/inst_slopes_L0_00_i1966_e90_Lw392.txt'
]

# Specify the file to be used for testing
test_traj_path = './Brutus data/plummer_triples_L0_00_i2025_e90_Lw392.csv'
test_slope_path = './data/inst_slopes_L0_00_i2025_e90_Lw392.txt'

# Identify position and velocity columns
pos_vel_cols = ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']

# Iterate over the training files and train the model incrementally
for i, (traj_path, slope_path) in enumerate(zip(train_trajectory_files, train_slope_files)):
    # Identify unique particles
    traj_df = pd.read_csv(traj_path)
    particles = traj_df[traj_df['Phase'].astype(int) == 1]['Particle Number'].unique()

    # Process the dataset
    X, y = process_dataset(traj_path, slope_path, pos_vel_cols, particles)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    history = mlp_model.fit(X_train, y_train, batch_size=32, epochs=1000, validation_data=(X_val, y_val))
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

# Load the test dataset
test_traj_df = pd.read_csv(test_traj_path)
particles = test_traj_df[test_traj_df['Phase'].astype(int) == 1]['Particle Number'].unique()

# Process the test dataset
X_test, y_test = process_dataset(test_traj_path, test_slope_path, pos_vel_cols, particles)

# Evaluate the model on the test set
y_pred = mlp_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Test MSE: {mse}')
print(f'Test R^2: {r2}')

# Plot true vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, alpha=0.5, label='True Values', color='blue')
plt.scatter(range(len(y_test)), y_pred, alpha=0.5, label='Predicted Values', color='red')
plt.title('True vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./figures/true_vs_predicted.png')
plt.show()

plt.figure(figsize=(12, 6))

# Plot histograms for true and predicted values
plt.hist(y_test, bins=30, alpha=0.5, label='True Values', color='blue', edgecolor='black')
plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted Values', color='red', edgecolor='black')

plt.title('Histogram of True vs Predicted Values')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./figures/true_vs_predicted_histogram.png')
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

# Compute correlation matrix
input_columns = [f'Particle {p} {col}' for p in particles for col in pos_vel_cols]
df_corr = pd.DataFrame(X, columns=input_columns)
df_corr['Target'] = y

correlation_matrix = df_corr.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
# select only the correlation between input features and target
# all rows except the last one and only the last column
sns.heatmap(correlation_matrix.iloc[:-1, -1:], annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix: Input Features vs. Target')
plt.xlabel('Target (Slope)')
plt.ylabel('Input Features')
plt.tight_layout()
plt.savefig('./figures/correlation_matrix.png')
plt.show()


""" # Example input: 18-dimensional coordinate (positions and velocities of 3 particles)
example_input = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  # Particle 1
                           0.7, 0.8, 0.9, 1.0, 1.1, 1.2,  # Particle 2
                           1.3, 1.4, 1.5, 1.6, 1.7, 1.8]]) # Particle 3

# Predict the instantaneous slope
predicted_slope = mlp_model.predict(example_input)
print(f'Predicted Instantaneous Slope: {predicted_slope[0][0]}') """