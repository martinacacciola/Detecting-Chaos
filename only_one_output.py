import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import seaborn as sns
import shap



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

        X.append(timestep_data.flatten())

        # Append Gaussian parameters as the target for this timestep
        #y.append(np.hstack([gaussian_params['mean'], gaussian_params['std'], gaussian_params['weight'], gaussian_params['height']]))
        y.append(gaussian_params['height'])

    X = np.array(X) # (n_timesteps, 18)
    y = np.array(y) # (n_timesteps, 1) using only one param

    return X, y


# Define the MLP model
# linear stack of layers

def create_mlp_model(input_shape):
    inputs = Input(shape=input_shape)

    # First hidden layer
    x = Dense(256, activation='swish')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second hidden layer
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third hidden layer
    x = Dense(64, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Fourth hidden layer
    x = Dense(32, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layers for each parameter with different activation functions
    #mean_output = Dense(2, activation='linear', name='mean_output')(x)
    #std_output = Dense(2, activation='softplus', name='std_output')(x) 
    #weight_output = Dense(2, activation='softmax', name='weight_output')(x)
    height_output = Dense(2, activation='softplus', name='height_output')(x)  # Softplus ensures positive output

    # Concatenate all outputs (8 - 4 for each of the 2 Gaussians)
    #output = Concatenate()([mean_output, std_output, weight_output, height_output])
    output = height_output
    
    # Create the final model
    final_model = Model(inputs=inputs, outputs=output)
    
    return final_model


    
input_shape = (9,)  # 3 positions for each of the 3 particles
mlp_model = create_mlp_model(input_shape)


history_list = []

# files for training and validation
train_trajectory_files = glob.glob('./Brutus data/*.csv')

train_gaussian_files = glob.glob('./data/*.txt')

# file for testing
test_traj_path = './Brutus data/test_data/plummer_triples_L0_00_i2025_e90_Lw392.csv'
test_gaussian_path = './data/test_data/gmm_parameters_L0_00_i2025_e90_Lw392.txt'

# Identify position and velocity columns
pos_vel_cols = ['X Position', 'Y Position', 'Z Position']

# below to concatenate all the files in one
# do it just once and save the concatenated files


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
#all_X = np.concatenate(all_X, axis=0)
all_y = np.concatenate(all_y, axis=0)
#np.savetxt('./data/all_X_decay.csv', all_X, delimiter=',')
#np.savetxt('./data/all_y_mean.csv', all_y, delimiter=',') 


# to normalize inputs and outputs (do it just once)
all_X = np.genfromtxt('./data/all_X_decay.csv', delimiter=',')
#all_y = np.genfromtxt('./data/all_y_decay.csv', delimiter=',')

# Normalize inputs and outputs
scaler_X = MinMaxScaler() #StandardScaler() 
all_X_tot = scaler_X.fit_transform(all_X)

# using minmax on the mean values
all_y = np.log(all_y)
scaler_y = MinMaxScaler()  #MinMaxScaler() #StandardScaler() 
# applying log to std values before doing the normalization
all_y_tot = scaler_y.fit_transform(all_y)

# Split the targets into separate groups (mean, std, weight, height) and normalize each separately
#all_y_split = np.split(all_y, 4, axis=-1)
#scalers_y = [MinMaxScaler().fit(y) for y in all_y_split]
#all_y_norm_split = [scaler.transform(y) for scaler, y in zip(scalers_y, all_y_split)]
#all_y_tot = np.column_stack(all_y_norm_split)


# save normalized data
#np.savetxt('./data/all_X_norm_decay.csv', all_X_tot, delimiter=',')
#np.savetxt('./data/all_y_norm_decay.csv', all_y_tot, delimiter=',') 

#using already normalized inputs and outputs # uncomment and comment part above later
#all_X_tot = np.genfromtxt('./data/all_X_norm_decay.csv', delimiter=',')
#all_y_tot = np.genfromtxt('./data/all_y_norm_decay.csv', delimiter=',')

# Shuffle the dataset
all_X_tot, all_y_tot = shuffle(all_X_tot, all_y_tot, random_state=42)

# divide the dataset into two sets: training+validation and test
train_val_percentage = 0.8
test_percentage = 0.2
train_val_size = int(train_val_percentage * len(all_X_tot))
test_size = len(all_X_tot) - train_val_size
train_val_X = all_X_tot[:train_val_size]
train_val_y = all_y_tot[:train_val_size]
test_X = all_X_tot[train_val_size:]
test_y = all_y_tot[train_val_size:]

# select a percentage of the data for training
sample_size = int(1 * len(train_val_X))
all_X, all_y = train_val_X[:sample_size], train_val_y[:sample_size]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

# Initialize dictionaries to store losses
losses_per_mean = {'train_loss': [], 'val_loss': []}

# Create the optimizer with the schedule
optimizer = Adam(learning_rate=0.01, clipvalue=0.5)

# Compile the model
mlp_model.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
)

# Training loop
batch_size = 32 #32
n_epochs = 1000 #4000
for epoch in range(n_epochs):
    # Shuffle the training data
    print(f'Epoch {epoch + 1}/{n_epochs}:')
    X_train, y_train = shuffle(X_train, y_train, random_state=epoch)
    
    # Select a random sample
    idx = np.random.choice(len(X_train), batch_size, replace=False)
    X_batch = X_train[idx] # (batch_size, 18)
    y_batch = y_train[idx] # (batch_size, 8)

    # Get current learning rate
    #current_lr = lr_schedule(optimizer.iterations)
    #print(f"Current learning rate: {float(current_lr):.6f}")
    
    # Train on the random sample
    mlp_model.fit(X_batch, y_batch,  epochs=1, verbose=1, validation_data=(X_val, y_val))
    
   # Predict training and validation losses separately for each parameter
    y_train_pred = mlp_model.predict(X_train, verbose=0) 
    y_val_pred = mlp_model.predict(X_val, verbose=0) 

    # compute loss on the single parameter
    train_loss = np.mean((y_train - y_train_pred) ** 2)
    val_loss = np.mean((y_val - y_val_pred) ** 2)

    losses_per_mean['train_loss'].append(train_loss)
    losses_per_mean['val_loss'].append(val_loss)
    

# Plot the loss evolution for the mean value
plt.figure(figsize=(10, 6))

plt.plot(losses_per_mean['train_loss'], label='Train Loss')
plt.plot(losses_per_mean['val_loss'], label='Validation Loss')
plt.title('Loss Evolution for Height')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.yscale('log')
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('./figures/only_loss_evolution_per_height.png')
plt.show()

### TEST
# select a percentage of the data for testing
sample_size = int(0.1 * len(train_val_X))
random_indices = np.random.choice(test_X.shape[0], sample_size, replace=False)
test_X, test_y = test_X[random_indices], test_y[random_indices]
#test_X, test_y = test_X[:sample_size], test_y[:sample_size]

# Number of random samples to test
num_random_samples = 100

# Lists to store real and predicted values for scatter plot
real_values = []
predicted_values = []

# Repeat the process for multiple random samples
for i in range(num_random_samples):
    # Generate a random index
    random_index = np.random.randint(0, test_X.shape[0])
    X_random = test_X[random_index].reshape(1, -1)
    y_random = test_y[random_index].reshape(1, -1)

    # Predict the random sample
    y_random_pred = mlp_model.predict(X_random)
    
    # Store real and predicted values for scatter plot
    real_values.append(y_random.flatten())
    predicted_values.append(y_random_pred.flatten())

    # Print the real and predicted values
    print(f"Iteration {i+1}:")
    print(f"Height - Real: {y_random.flatten()}, Predicted: {y_random_pred.flatten()}")

# Convert the lists of real and predicted values to numpy arrays
#real_values_array = np.array(real_values).flatten()
#predicted_values_array = np.array(predicted_values).flatten()

# Reshape the arrays to match the expected shape for the scaler
#real_values_array = real_values_array.reshape(-1, 1)
#predicted_values_array = predicted_values_array.reshape(-1, 1)

# Denormalize using the scaler
real_values_denorm = scaler_y.inverse_transform(real_values).flatten()
predicted_values_denorm = scaler_y.inverse_transform(predicted_values).flatten()

# Scatter plot for actual vs predicted values
plt.figure(figsize=(10, 10))
plt.scatter(real_values_denorm, predicted_values_denorm, c='orange', alpha=0.5, label='Predicted')
plt.scatter(real_values_denorm, real_values_denorm, c='blue', alpha=0.5, label='True')
plt.xlabel('True Height', fontsize=15)
plt.ylabel('Predicted Height', fontsize=15)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.savefig('./figures/height_actual_vs_predicted.png')
plt.show()

# Residual plot
plt.figure(figsize=(10, 10))
residuals = real_values_denorm - predicted_values_denorm
plt.scatter(real_values_denorm, residuals, c='green', alpha=0.5, label='Residuals')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('True Height', fontsize=15)
plt.ylabel('Residuals Height', fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig('./figures/height_residuals_plot.png')
plt.show()



def plot_cdf_and_distributions(true_values, predicted_values):
    # Calculate absolute errors
    errors = np.abs(true_values - predicted_values)
    
    # Plot CDF of errors
    sorted_errors = np.sort(errors)
    cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(sorted_errors, cdf, label='CDF of Prediction Errors')
    plt.axvline(x=1, color='r', linestyle='--', label='1 Order of Magnitude')
    plt.xlabel('Absolute Error')
    plt.ylabel('CDF')
    plt.title('CDF of Prediction Errors')
    plt.legend()
    
    # Plot distributions of true and predicted values
    plt.subplot(1, 2, 2)
    plt.hist(true_values, bins=30, alpha=0.5, label='True Values', density=True)
    plt.hist(predicted_values, bins=30, alpha=0.5, label='Predicted Values', density=True)
    
    # Fit a normal distribution to the data
    mu_true, std_true = norm.fit(true_values)
    mu_pred, std_pred = norm.fit(predicted_values)
    
    # Plot the PDF of the fitted normal distributions
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p_true = norm.pdf(x, mu_true, std_true)
    p_pred = norm.pdf(x, mu_pred, std_pred)
    
    plt.plot(x, p_true, 'k', linewidth=2, label='True Values Fit')
    plt.plot(x, p_pred, 'r', linewidth=2, label='Predicted Values Fit')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of True and Predicted Values')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage:
# true_values = np.array([...])  # Replace with your true values
# predicted_values = np.array([...])  # Replace with your predicted values
plot_cdf_and_distributions(real_values_denorm, predicted_values_denorm)
