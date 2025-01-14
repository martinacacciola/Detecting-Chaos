import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
import seaborn as sns

# TODO: 
# problema: custom_loss (o anche l'altra) non è adatta agli split e non la si sta utilizzando per i grafici
# !! understand if it is timestep wise
# 1) change the input s.t. 1 particle is always at the origin, the 2 on x axis and the 3 rotated accordingly
# ok but should we put less inputs in this case?
# 2) do not process the whole trajectory when training - select only a subset of timesteps 
# (random coordinates in different points from same trajectory)
# 3) use a more appropriate loss function 

# the network is learning from one coordinate at a time
# to each point belonging uniquely to a trajectory, map the unique parameters
# goal: from one coordinate, predict the lyapunov exponent distr of the whole trajectory

def custom_loss(y_true, y_pred):
    # Split predictions into 4 groups (mean, std, weight, height) along the last dimension of the tensor
    y_true_split = tf.split(y_true, num_or_size_splits=4, axis=-1)
    y_pred_split = tf.split(y_pred, num_or_size_splits=4, axis=-1)
    
    # Compute MSE for each group
    mse_per_param = [tf.reduce_mean(tf.square(y_t - y_p)) for y_t, y_p in zip(y_true_split, y_pred_split)]
    
    # Return the mean of all parameters losses
    return tf.reduce_mean(mse_per_param)


def process_dataset(traj_path, gaussian_path, pos_vel_cols, particles):
    # Load the dataset
    traj_df = pd.read_csv(traj_path)
    gaussian_df = pd.read_csv(gaussian_path, header=None, sep='\s+')
    gaussian_params = {
        'mean': gaussian_df.iloc[0].values.astype(float),
        'std': gaussian_df.iloc[1].values.astype(float),
        'weight': gaussian_df.iloc[2].values.astype(float),
        'height': gaussian_df.iloc[3].values.astype(float),
    }

    forward_trajectory = traj_df[traj_df['Phase'].astype(int) == 1]
    timesteps = forward_trajectory['Timestep'].unique()

    # X = 18 coordinates for all particles combined
    # y = 8 target parameters (mean, std, weight, height) for each timestep
    X = []
    y = []

    # Loop through each timestep
    for t in timesteps:
        timestep_data = []
        for p in particles:
            forward_state = forward_trajectory[(forward_trajectory['Particle Number'] == p) & (forward_trajectory['Timestep'] == t)]
            
            # Combine position and velocity into a single array
            pos_vel = forward_state[pos_vel_cols].values[0]
            timestep_data.append(pos_vel)  # Add particle's data for this timestep

        timestep_data = np.array(timestep_data)

        """ # Reorganize particles
        # assume first particle to be at the origin, second on x axis and third on x-y plane
        origin_particle = timestep_data[0, :2]  # select x-y values
        # euclidean distances from origin particles to all the others
        distances = np.linalg.norm(timestep_data[:, :2] - origin_particle, axis=1)
        nearest_neighbor_idx = np.argmin(distances[1:]) + 1 

        # Translate particles so that the origin particle is at the origin
        # subtract the coords of origin particle from all the others
        timestep_data[:, :2] -= origin_particle

        # Calculate the angle to rotate the nearest neighbor to the positive x-axis
        nearest_neighbor = timestep_data[nearest_neighbor_idx, :2]
        angle = -np.arctan2(nearest_neighbor[1], nearest_neighbor[0])

        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # Rotate all particles
        timestep_data[:, :2] = np.dot(timestep_data[:, :2], rotation_matrix) """

        X.append(timestep_data.flatten())

        # Append Gaussian parameters as the target for this timestep
        y.append(np.hstack([gaussian_params['mean'], gaussian_params['std'], gaussian_params['weight'], gaussian_params['height']]))

    X = np.array(X) # (n_timesteps, 18)
    y = np.array(y) # (n_timesteps, 8)

    return X, y


# Define the MLP model
# linear stack of layers

def create_mlp_model(input_shape):
    inputs = Input(shape=input_shape)

    # First hidden layer
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second hidden layer
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third hidden layer
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Fourth hidden layer
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Fifth hidden layer
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layers for each parameter with different activation functions
    mean_output = Dense(2, activation='linear', name='mean_output')(x)
    std_output = Dense(2, activation='softplus', name='std_output')(x)
    weight_output = Dense(2, activation='softmax', name='weight_output')(x)
    height_output = Dense(2, activation='softplus', name='height_output')(x)  # Softplus ensures positive output

    # Concatenate all outputs (8 - 4 for each of the 2 Gaussians)
    output = Concatenate()([mean_output, std_output, weight_output, height_output])
    
    # Create the final model
    final_model = Model(inputs=inputs, outputs=output)
    
    return final_model

def leaky_mlp_model(input_shape):
    inputs = Input(shape=input_shape)

    # First hidden block
    x = Dense(512)(inputs)  # Increase neurons
    x = LeakyReLU(negative_slope=0.1)(x)  # Use LeakyReLU instead of ReLU for better gradient flow
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Increase dropout rate

    # Second hidden block
    x = Dense(512)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Third hidden block
    x = Dense(256)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Fourth hidden block
    x = Dense(256)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Fifth hidden block
    x = Dense(128)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Sixth hidden block
    x = Dense(128)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Output layers for each parameter with different activation functions
    mean_output = Dense(2, activation='linear', name='mean_output')(x)
    std_output = Dense(2, activation='softplus', name='std_output')(x)
    weight_output = Dense(2, activation='softmax', name='weight_output')(x)
    height_output = Dense(2, activation='softplus', name='height_output')(x)  # Softplus ensures positive output

    # Concatenate all outputs (8 - 4 for each of the 2 Gaussians)
    output = Concatenate()([mean_output, std_output, weight_output, height_output])
    
    # Create the final model
    final_model = Model(inputs=inputs, outputs=output)
    
    return final_model


input_shape = (18,)  # 6 for each of the 3 particles
mlp_model = create_mlp_model(input_shape)

# Compile the model
# default learning rate = 0.001
mlp_model.compile(
    optimizer=Adam(),   #learning_rate=0.01
    loss= 'mean_squared_error',   #custom_loss,
    metrics=['mae']
)
# Print the model summary to verify the input shape
#mlp_model.summary()

history_list = []

# files for training and validation
train_trajectory_files = glob.glob('./Brutus data/*.csv')

train_gaussian_files = glob.glob('./data/*.txt')

# file for testing
test_traj_path = './Brutus data/test_data/plummer_triples_L0_00_i2025_e90_Lw392.csv'
test_gaussian_path = './data/test_data/gmm_parameters_L0_00_i2025_e90_Lw392.txt'

# Identify position and velocity columns
pos_vel_cols = ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']

# below to concatenate all the files in one
# do it just once and save the concatenated files
""" 

# Initialize lists to hold all trajectories and parameters
all_X = []
all_y = []

# Load all trajectories and parameters
for traj_path, gaussian_path in zip(train_trajectory_files, train_gaussian_files):
    traj_df = pd.read_csv(traj_path)
    particles = traj_df[traj_df['Phase'].astype(int) == 1]['Particle Number'].unique()
    
    X, y = process_dataset(traj_path, gaussian_path, pos_vel_cols, particles)
    all_X.append(X)
    all_y.append(y)

# Concatenate all trajectories and parameters
all_X = np.concatenate(all_X, axis=0)
all_y = np.concatenate(all_y, axis=0)
np.savetxt('./data/all_X.csv', all_X, delimiter=',')
np.savetxt('./data/all_y.csv', all_y, delimiter=',')  """

all_X_tot = np.genfromtxt('./data/all_X.csv', delimiter=',')
all_y_tot = np.genfromtxt('./data/all_y.csv', delimiter=',')

# Shuffle the dataset
all_X_tot, all_y_tot = shuffle(all_X_tot, all_y_tot, random_state=42)

print('All X size:', all_X_tot.shape)
print('All y size:', all_y_tot.shape)

# select a percentage of the data for training
sample_size = int(0.1 * len(all_X_tot))
all_X, all_y = all_X_tot[:sample_size], all_y_tot[:sample_size]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
print('X_train size:', X_train.shape)
print('y_train size:', y_train.shape)
print('X_val size:', X_val.shape)
print('y_val size:', y_val.shape)

## added
# Ensure no overlap by checking indices
# set = unordered collection of unique elements
train_indices = set(map(tuple, X_train))  # Convert to set of tuples for comparison
val_indices = set(map(tuple, X_val))

overlap = train_indices.intersection(val_indices)
if overlap:
    print(f"Overlap detected! Overlapping entries: {overlap}")
else:
    print("No overlap between training and validation sets.")
##

# Initialize dictionaries to store losses
losses_per_param = {'train_loss': {param: [] for param in ['mean', 'std', 'weight', 'height']},
                    'val_loss': {param: [] for param in ['mean', 'std', 'weight', 'height']}}

# Trainining loop
batch_size = 32
n_epochs = 10000 #4000
for epoch in range(n_epochs):
    # Shuffle the training data
    print(f'Epoch {epoch + 1}/{n_epochs}:')
    X_train, y_train = shuffle(X_train, y_train, random_state=epoch)
    
    # Select a random sample
    idx = np.random.choice(len(X_train), batch_size, replace=False)
    X_batch = X_train[idx] # (batch_size, 18)
    y_batch = y_train[idx] # (batch_size, 8)
    
    # Train on the random sample
    mlp_model.fit(X_batch, y_batch, epochs=1, verbose=1, validation_data=(X_val, y_val))
    
    # Predict training and validation losses separately for each parameter
    # should we use the whole dataset or just the batch? when using just the batch oscillatory loss for training
    y_train_pred = mlp_model.predict(X_train, verbose=0) 
    y_val_pred = mlp_model.predict(X_val, verbose=0) 
    
    # Split true and predicted into separate groups
    y_train_split = np.split(y_train, 4, axis=-1)
    y_val_split = np.split(y_val, 4, axis=-1)
    y_train_pred_split = np.split(y_train_pred, 4, axis=-1)
    y_val_pred_split = np.split(y_val_pred, 4, axis=-1)
    
    # Compute and record losses for each parameter
    for i, param in enumerate(['mean', 'std', 'weight', 'height']):
        train_loss = np.mean((y_train_split[i] - y_train_pred_split[i]) ** 2)
        val_loss = np.mean((y_val_split[i] - y_val_pred_split[i]) ** 2)
        losses_per_param['train_loss'][param].append(train_loss)
        losses_per_param['val_loss'][param].append(val_loss)

# Plot the loss evolution for each parameter
plt.figure(figsize=(14, 8))

for idx, param in enumerate(['mean', 'std', 'weight', 'height']):
    plt.subplot(2, 2, idx + 1)
    plt.plot(losses_per_param['train_loss'][param], label='Train Loss')
    plt.plot(losses_per_param['val_loss'][param], label='Validation Loss')
    plt.title(f'Loss Evolution for {param.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    #plt.yscale('log')
    plt.legend()
    plt.tight_layout()

# Save the figure
plt.savefig('./figures/loss_evolution_per_param.png')
plt.show()

# Evaluate on the test dataset
test_traj_df = pd.read_csv(test_traj_path)
particles = test_traj_df[test_traj_df['Phase'].astype(int) == 1]['Particle Number'].unique()
X_test, y_test = process_dataset(test_traj_path, test_gaussian_path, pos_vel_cols, particles)

# Predict test results
y_pred = mlp_model.predict(X_test)
y_test_split = np.split(y_test, 4, axis=-1)
y_pred_split = np.split(y_pred, 4, axis=-1)

# Compute test losses for each parameter
test_losses = {param: mean_squared_error(y_test_split[idx], y_pred_split[idx]) 
               for idx, param in enumerate(['mean', 'std', 'weight', 'height'])}

print("Test Losses (MSE) per Parameter:")
for param, loss in test_losses.items():
    print(f"{param.capitalize()}: {loss:.4f}")

# Test performance with one random set of coordinates from one timestep
# generate index btw 0 and number of samples
random_index = np.random.randint(0, X_test.shape[0])
X_random = X_test[random_index].reshape(1, -1)
y_random = y_test[random_index].reshape(1, -1)
print('Random X size:', X_random.shape) # 1, 18
print('Random y size:', y_random.shape) # 1, 8

y_random_pred = mlp_model.predict(X_random)
y_random_split = np.split(y_random, 4, axis=-1)
y_random_pred_split = np.split(y_random_pred, 4, axis=-1)

random_losses = {param: mean_squared_error(y_random_split[idx], y_random_pred_split[idx]) 
                 for idx, param in enumerate(['mean', 'std', 'weight', 'height'])}

# Print random sample losses
print("\nRandom Sample Losses (MSE) per Parameter:")
for param, loss in random_losses.items():
    print(f"{param.capitalize()}: {loss:.4f}")


""" # da sistemare
# Summary plot for average loss across all files for each parameter
for param in history_per_param:
    train_losses = np.array(history_per_param[param]['train_loss'])
    val_losses = np.array(history_per_param[param]['val_loss'])

    if train_losses.size > 0 and val_losses.size > 0:
        mean_train_loss = np.mean(train_losses, axis=0)
        std_train_loss = np.std(train_losses, axis=0)
        mean_val_loss = np.mean(val_losses, axis=0)
        std_val_loss = np.std(val_losses, axis=0)

        epochs = range(1, len(mean_train_loss) + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, mean_train_loss, label=f'Mean Training Loss ({param})')
        plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.2)
        plt.plot(epochs, mean_val_loss, label=f'Mean Validation Loss ({param})')
        plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.2)
        plt.title(f'Average Model Loss Across All Files ({param})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./figures/average_loss_{param}.png')
        plt.show()
    else:
        print(f"No valid training or validation loss data available for {param}.") """

""" # Compute correlation matrix
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
plt.show() """


""" # Example input: 18-dimensional coordinate (positions and velocities of 3 particles)
example_input = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  # Particle 1
                           0.7, 0.8, 0.9, 1.0, 1.1, 1.2,  # Particle 2
                           1.3, 1.4, 1.5, 1.6, 1.7, 1.8]]) # Particle 3

# Predict the instantaneous slope
predicted_slope = mlp_model.predict(example_input)
print(f'Predicted Instantaneous Slope: {predicted_slope[0][0]}') """

""" 
# previous model
def create_mlp_model(input_shape):
    inputs = Input(shape=input_shape)

    # First hidden layer
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second hidden layer
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third hidden layer
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Fourth hidden layer
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Fifth hidden layer
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layers for each parameter with different activation functions
    mean_output = Dense(2, activation='linear', name='mean_output')(x)
    std_output = Dense(2, activation='softplus', name='std_output')(x)
    weight_output = Dense(2, activation='softmax', name='weight_output')(x)
    height_output = Dense(2, activation='softplus', name='height_output')(x)  # Softplus ensures positive output

    # Concatenate all outputs (8 - 4 for each of the 2 Gaussians)
    output = Concatenate()([mean_output, std_output, weight_output, height_output])
    
    # Create the final model
    final_model = Model(inputs=inputs, outputs=output)
    
    return final_model """