import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import seaborn as sns
import shap


def custom_loss(y_true, y_pred):
    # Split predictions into 4 groups (mean, std, weight, height) along the last dimension of the tensor
    y_true_split = tf.split(y_true, num_or_size_splits=4, axis=-1)
    y_pred_split = tf.split(y_pred, num_or_size_splits=4, axis=-1)
    
    # Define weights for each parameter
    weights = [0.3333167, 0.3333167, 0.00005, 0.3333167]  # Adjust these weights as needed
    
    # Compute MSE for each group
    mse_per_param = [tf.reduce_mean(tf.square(y_t - y_p)) for y_t, y_p in zip(y_true_split, y_pred_split)]
    
    # Compute weighted sum of MSEs
    weighted_mse = tf.reduce_sum([w * mse for w, mse in zip(weights, mse_per_param)])
    
    return weighted_mse


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
        y.append(np.hstack([gaussian_params['mean'], gaussian_params['std'], gaussian_params['weight'], gaussian_params['height']]))

    X = np.array(X) # (n_timesteps, 18)
    y = np.array(y) # (n_timesteps, 8)

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
    mean_output = Dense(2, activation='linear', name='mean_output')(x)
    std_output = Dense(2, activation='softplus', name='std_output')(x)
    weight_output = Dense(2, activation='softmax', name='weight_output')(x)
    height_output = Dense(2, activation='softplus', name='height_output')(x)  # Softplus ensures positive output

    # Concatenate all outputs (8 - 4 for each of the 2 Gaussians)
    output = Concatenate()([mean_output, std_output, weight_output, height_output])
    
    # Create the final model
    final_model = Model(inputs=inputs, outputs=output)
    
    return final_model


input_shape = (9,)  # 6 for each of the 3 particles
mlp_model = create_mlp_model(input_shape)


""" 
optimizer = Adam(learning_rate=0.01, clipvalue=0.5)  

# Compile the model
# default learning rate = 0.001
mlp_model.compile(
    optimizer= optimizer,     # SGD(learning_rate=0.01),   #learning_rate=0.01
    loss= custom_loss #'mean_squared_error',  
    #metrics=['mae']
) """


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
#all_y = np.concatenate(all_y, axis=0)
#np.savetxt('./data/all_X_decay.csv', all_X, delimiter=',')
#np.savetxt('./data/all_y_decay.csv', all_y, delimiter=',') 


# to normalize inputs and outputs (do it just once)
all_X = np.genfromtxt('./data/all_X_decay.csv', delimiter=',')
all_y = np.genfromtxt('./data/all_y_decay.csv', delimiter=',')

# Normalize inputs and outputs
scaler_X = MinMaxScaler() #StandardScaler()
all_X_tot = scaler_X.fit_transform(all_X)

# Split the targets into separate groups (mean, std, weight, height) and normalize each separately
all_y_split = np.split(all_y, 4, axis=-1)
scalers_y = [MinMaxScaler().fit(y) for y in all_y_split]
all_y_norm_split = [scaler.transform(y) for scaler, y in zip(scalers_y, all_y_split)]
all_y_tot = np.column_stack(all_y_norm_split)


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
losses_per_param = {'train_loss': {param: [] for param in ['mean', 'std', 'weight', 'height']},
                    'val_loss': {param: [] for param in ['mean', 'std', 'weight', 'height']}}

## ADDED
batch_size = 32 
# Define the learning rate schedule
""" initial_learning_rate = 0.1
decay_steps = 500 #len(X_train) // batch_size  # number of steps after which the learning rate decays
decay_rate = 0.9  # at each decay step, the learning rate is reduced by this factor
min_learning_rate = 1e-7 # Minimum learning rate

lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True,
    name="ExponentialDecay"
) """

# Create the optimizer with the schedule
optimizer = Adam(learning_rate=0.001, clipvalue=0.5)

# Compile the model
mlp_model.compile(
    optimizer=optimizer,
    loss=custom_loss
)

# Training loop
#batch_size = 32 #32
n_epochs = 2000 #4000
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
    
    # Train on the random sample
    #mlp_model.fit(X_batch, y_batch, epochs=1, verbose=1, validation_data=(X_val, y_val))
    
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
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

# Save the figure
plt.savefig('./figures/loss_evolution_per_param.png')
plt.show()


### TEST
# select a percentage of the data for testing
sample_size = int(0.1 * len(train_val_X))
random_indices = np.random.choice(test_X.shape[0], sample_size, replace=False)
test_X, test_y = test_X[random_indices], test_y[random_indices]
#test_X, test_y = test_X[:sample_size], test_y[:sample_size]

# Number of random samples to test
num_random_samples = 50

# Lists to store real and predicted values for scatter plot
real_values = {param: [] for param in ['mean', 'std', 'weight', 'height']}
predicted_values = {param: [] for param in ['mean', 'std', 'weight', 'height']}

# Repeat the process for multiple random samples
for i in range(num_random_samples):
    # Generate a random index
    random_index = np.random.randint(0, test_X.shape[0])
    X_random = test_X[random_index].reshape(1, -1)
    y_random = test_y[random_index].reshape(1, -1)

    # Predict the random sample
    y_random_pred = mlp_model.predict(X_random)
    
    # Split the real and predicted values
    y_random_split = np.split(y_random, 4, axis=-1)
    y_random_pred_split = np.split(y_random_pred, 4, axis=-1)

    # Store real and predicted values for scatter plot
    for idx, param in enumerate(['mean', 'std', 'weight', 'height']):
        real_values[param].append(y_random_split[idx].flatten())
        predicted_values[param].append(y_random_pred_split[idx].flatten())

    # Print the real and predicted values for each parameter
    print(f"Iteration {i+1}:")
    for idx, param in enumerate(['mean', 'std', 'weight', 'height']):
        print(f"{param.capitalize()} - Real: {y_random_split[idx].flatten()}, Predicted: {y_random_pred_split[idx].flatten()}")

# Convert the lists of real and predicted values to numpy arrays
real_values_array = np.column_stack([np.concatenate(real_values[param]) for param in ['mean', 'std', 'weight', 'height']])
predicted_values_array = np.column_stack([np.concatenate(predicted_values[param]) for param in ['mean', 'std', 'weight', 'height']])

# Ensure correct splitting of real and predicted values into groups of 2 (for each Gaussian)
real_values_split = [real_values_array[:, i:i + 2] for i in range(0, real_values_array.shape[1], 2)]
predicted_values_split = [predicted_values_array[:, i:i + 2] for i in range(0, predicted_values_array.shape[1], 2)]

# Denormalize using the scalers, ensuring correct shapes for inverse_transform
real_values_denorm = np.column_stack([scaler.inverse_transform(real) for scaler, real in zip(scalers_y, real_values_split)])
predicted_values_denorm = np.column_stack([scaler.inverse_transform(pred) for scaler, pred in zip(scalers_y, predicted_values_split)])

##
# Scatter plot for actual vs predicted values
""" plt.figure(figsize=(14, 8))

for idx, param in enumerate(['mean', 'std', 'weight', 'height']):
    plt.subplot(2, 2, idx + 1)
    plt.scatter(real_values_denorm[:, idx], predicted_values_denorm[:, idx], alpha=0.5, label='Predicted vs Actual')
    plt.scatter(real_values_denorm[:, idx], real_values_denorm[:, idx], alpha=0.5, label='Actual vs Actual')
    #plt.scatter(predicted_values_denorm[:, idx], predicted_values_denorm[:, idx], alpha=0.5, label='Predicted vs Predicted')
    plt.title(f'Actual vs Predicted for {param.capitalize()}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.tight_layout()

# Save the figure
#plt.savefig('./figures/actual_vs_predicted.png')
plt.show()
## """
#provare ad usare i valori non denormalizzati


plt.figure(figsize=(10,10))
for idx, param in enumerate(['mean', 'std', 'weight', 'height']):
    plt.subplot(2, 2, idx+1)
    plt.scatter(real_values_denorm[:, idx], predicted_values_denorm[:, idx], c='orange', alpha=0.5, label='Predicted') #crimson
    plt.scatter(real_values_denorm[:, idx], real_values_denorm[:, idx], c='blue', alpha=0.5, label='True')
    
    #p1 = max(predicted_values_denorm[:, idx]), max(real_values_denorm[:, idx])
    #p2 = min(predicted_values_denorm[:, idx]), min(real_values_denorm[:, idx])
    #plt.plot([p1, p2], [p1, p2], 'b-', label='Identity line')
    plt.xlabel(f'True {param}', fontsize=15)
    plt.ylabel(f'Predicted {param}', fontsize=15)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
plt.savefig('./figures/actual_vs_predicted.png')
plt.show()

#Residual plot
plt.figure(figsize=(10,10))
for idx, param in enumerate(['mean', 'std', 'weight', 'height']):
    plt.subplot(2, 2, idx+1)
    residuals = real_values_denorm[:, idx] - predicted_values_denorm[:, idx]
    plt.scatter(real_values_denorm[:, idx], residuals, c='green', alpha=0.5, label='Residuals')
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel(f'True {param}', fontsize=15)
    plt.ylabel(f'Residuals {param}', fontsize=15)
    plt.legend()
    plt.tight_layout()
plt.savefig('./figures/residuals_plot.png')
plt.show()

""" # ---- SHAP integration ----
# Define base feature names for one particle
pos_vel_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']

# Generate feature names for all particles
num_particles = 3
feature_names = [f'{col}_particle_{i+1}' for i in range(num_particles) for col in pos_vel_cols]

# Define a small subset of test data to calculate SHAP values
shap_sample_size = 100 
shap_indices = np.random.choice(test_X.shape[0], shap_sample_size, replace=False)
shap_test_X = test_X[shap_indices]

# Initialize SHAP explainer
explainer = shap.Explainer(mlp_model, shap_test_X)

# Compute SHAP values for the test data
shap_values = explainer.shap_values(shap_test_X)

# Print shapes for debugging
#print(f"Shape of shap_values: {shap_values.shape}")
#print(f"Shape of shap_test_X: {shap_test_X.shape}")
#print(f"Length of feature_names: {len(feature_names)}")

# Split SHAP values by output features
output_features = ['mean', 'std', 'weight', 'height']

# Plot SHAP summary for each output feature
for i, feature_name in enumerate(output_features):
    print(f"Creating SHAP summary plot for {feature_name.capitalize()}")
    shap.summary_plot(shap_values[:, :, i], shap_test_X, feature_names=feature_names, show=False, plot_type='violin')
    plt.title(f"SHAP Summary for {feature_name.capitalize()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'./figures/shap_summary_{feature_name}.png')
    plt.show() """








































""" # Predicted vs Actual scatter plot
for idx, param in enumerate(['mean', 'std', 'weight', 'height']):
    plt.figure(figsize=(10, 5))
    sns.regplot(x=real_values_denorm[:, idx], y=predicted_values_denorm[:, idx], scatter_kws={'color':'blue', 'alpha':0.5}, line_kws={'color':'red'}, label='Actual vs Predicted')
    plt.xlabel(f'Actual {param}')
    plt.ylabel(f'Predicted {param}')
    plt.title(f'Predicted vs Actual Scatter Plot for {param}')
    plt.legend()
    plt.savefig(f'./figures/predicted_vs_actual_{param}.png')
    plt.show()

# Residual plot
for idx, param in enumerate(['mean', 'std', 'weight', 'height']):
    residuals = real_values_denorm[:, idx] - predicted_values_denorm[:, idx]

    plt.figure(figsize=(10, 5))
    sns.regplot(x=predicted_values_denorm[:, idx], y=residuals, scatter_kws={'color':'blue', 'alpha':0.5}, line_kws={'color':'red'}, label='Residuals')
    plt.xlabel(f'Predicted {param}')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {param}')
    plt.legend()
    plt.savefig(f'./figures/residual_plot_{param}.png')
    plt.show() """


""" 
# Predict on the test set
y_test_pred = mlp_model.predict(test_X)

# Denormalize true and predicted values
y_test_denorm = scaler.inverse_transform(test_y)
y_test_pred_denorm = scaler.inverse_transform(y_test_pred)

# Generate scatter plots for each parameter
parameters = ['mean', 'std', 'weight', 'height']
for idx, param in enumerate(parameters):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test_denorm[:, idx], y_test_pred_denorm[:, idx], alpha=0.6, edgecolor='k', label='Predicted')
    plt.plot([min(y_test_denorm[:, idx]), max(y_test_denorm[:, idx])],
             [min(y_test_denorm[:, idx]), max(y_test_denorm[:, idx])], 'r--', linewidth=2, label='Ideal')
    plt.title(f'True vs Predicted: {param}', fontsize=14)
    plt.xlabel(f'True {param}', fontsize=12)
    plt.ylabel(f'Predicted {param}', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'test_true_vs_predicted_{param}.png')
    plt.close()



"""


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
    
    return final_model 
    
    
    # leaky model
    # def leaky_mlp_model(input_shape):
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
    
    return final_model"""