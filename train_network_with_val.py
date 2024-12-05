import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import layers, models, losses, optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from network import preprocess_trajectory, build_phase_space_model

# TODO: inserire piu sample come input
# come parametro al posto dei pesi usare l'altezza relativa

# total loss function (sum of the three losses)
def custom_gmm_loss(y_true, y_pred):
    """
    Custom GMM loss function for concatenated outputs.

    Args:
        y_true (tensor): True parameter values, concatenated [means, stds, weights].
        y_pred (tensor): Predicted values, concatenated [means, stds, weights].

    Returns:
        Tensor: Combined loss value = sum of mean_loss, std_loss, and weight_loss.
    """

    # Split y_true and y_pred into means, stds, and weights
    means_true, stds_true, weights_true = tf.split(y_true, num_or_size_splits=3, axis=1)
    means_pred, stds_pred, weights_pred = tf.split(y_pred, num_or_size_splits=3, axis=1)

    # losses as MSE
    mean_loss = tf.reduce_mean(tf.square(means_true - means_pred))
    std_loss = tf.reduce_mean(tf.square(stds_true - stds_pred))
    weight_loss = tf.reduce_mean(losses.categorical_crossentropy(weights_true, weights_pred))

    return mean_loss + std_loss + weight_loss

# Loss functions for individual components
# MSE btw true and predicted parameters
def mean_loss_metric(y_true, y_pred):
    means_true, _, _ = tf.split(y_true, num_or_size_splits=3, axis=1)
    means_pred, _, _ = tf.split(y_pred, num_or_size_splits=3, axis=1)
    return tf.reduce_mean(tf.square(means_true - means_pred))

def std_loss_metric(y_true, y_pred):
    _, stds_true, _ = tf.split(y_true, num_or_size_splits=3, axis=1)
    _, stds_pred, _ = tf.split(y_pred, num_or_size_splits=3, axis=1)
    return tf.reduce_mean(tf.square(stds_true - stds_pred))

# mean of categorical cross-entropy
def weight_loss_metric(y_true, y_pred):
    _, _, weights_true = tf.split(y_true, num_or_size_splits=3, axis=1)
    _, _, weights_pred = tf.split(y_pred, num_or_size_splits=3, axis=1)
    return tf.reduce_mean(losses.categorical_crossentropy(weights_true, weights_pred))


def process_file_pair(trajectory_file, gmm_params_file):
    # Preprocess trajectories and GMM parameters
    forward_trajectories, backward_trajectories, gaussian_fit_params = preprocess_trajectory(
        file_path=trajectory_file,
        gaussian_fit_params_path=gmm_params_file
    )
    
    # Convert trajectories to NumPy arrays
    forward_inputs = np.array(list(forward_trajectories.values()))
    backward_inputs = np.array(list(backward_trajectories.values()))
    
    # Tile GMM parameters to match trajectory data shape
    means_true = np.tile(gaussian_fit_params[0, :], (forward_inputs.shape[0], 1))
    stds_true = np.tile(gaussian_fit_params[1, :], (forward_inputs.shape[0], 1))
    weights_true = np.tile(gaussian_fit_params[2, :], (forward_inputs.shape[0], 1))
    y_true = np.concatenate([means_true, stds_true, weights_true], axis=1)
    
    return forward_inputs, backward_inputs, y_true


# Paths to trajectory and GMM parameter files
trajectory_files = sorted(glob.glob('./Brutus data/*.csv'))
gmm_params_files = sorted(glob.glob('./data/gmm_parameters*.txt'))

# Initialize lists to hold all data
all_forward_inputs, all_backward_inputs, all_y_true = [], [], []

for trajectory_file, gmm_params_file in zip(trajectory_files, gmm_params_files):
    forward_inputs, backward_inputs, y_true = process_file_pair(trajectory_file, gmm_params_file)
    all_forward_inputs.append(forward_inputs)
    all_backward_inputs.append(backward_inputs)
    all_y_true.append(y_true)

# Concatenate all data
combined_forward_inputs = np.concatenate(all_forward_inputs, axis=0)
combined_backward_inputs = np.concatenate(all_backward_inputs, axis=0)
combined_y_true = np.concatenate(all_y_true, axis=0)


# Shuffle the data
combined_forward_inputs, combined_backward_inputs, combined_y_true = shuffle(
    combined_forward_inputs, combined_backward_inputs, combined_y_true, random_state=42
)

# Split into training and validation sets
train_forward_inputs, val_forward_inputs, train_backward_inputs, val_backward_inputs, train_y_true, val_y_true = train_test_split(
    combined_forward_inputs, combined_backward_inputs, combined_y_true, test_size=0.2, random_state=42
)


""" ## Inputs
# Trajectories
forward_inputs = np.array(list(forward_trajectories.values()))
backward_inputs = np.array(list(backward_trajectories.values()))


# the first row of params is repeated across all rows of input array 
# to match the shape
means_true = np.tile(gaussian_fit_params[0, :], (forward_inputs.shape[0], 1))
stds_true = np.tile(gaussian_fit_params[1, :], (forward_inputs.shape[0], 1))
weights_true = np.tile(gaussian_fit_params[2, :], (forward_inputs.shape[0], 1))
# True parameters
y_true = np.concatenate([means_true, stds_true, weights_true], axis=1) 

# Split the data into training and validation sets (80% vs 20%)
train_forward_inputs, val_forward_inputs, train_backward_inputs, val_backward_inputs, train_y_true, val_y_true = train_test_split(
    forward_inputs, backward_inputs, y_true, test_size=0.2, random_state=42) """

sequence_length = forward_inputs.shape[1]
feature_dim = forward_inputs.shape[2]
# hyperparameters
latent_dim = 128
num_epochs = 100
batch_size = 32
num_components = 2 #means_true.shape[0]
#learning_rate = 0.01

# Build and compile the model
model = build_phase_space_model(sequence_length, feature_dim, latent_dim, num_components)

model.compile(
    optimizer=optimizers.Adam(),
    loss=custom_gmm_loss,
    metrics=[mean_loss_metric, std_loss_metric, weight_loss_metric],
)

# Train the model with validation data
history = model.fit(
    [train_forward_inputs, train_backward_inputs],
    train_y_true,
    validation_data=([val_forward_inputs, val_backward_inputs], val_y_true),
    epochs=num_epochs,
    batch_size=batch_size,
)

# Save history
# this will be used to do do hyperparameter tuning in the grid search
#filename = f'./saved_results/training_history_{num_epochs}epochs_{batch_size}batch.csv'
#pd.DataFrame(history.history).to_csv('./saved_results/training_history.csv', index=False)

# Plot the training and validation history
plt.figure(figsize=(12, 8))

# Plot loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot mean MSE
plt.subplot(2, 2, 2)
plt.plot(history.history['mean_loss_metric'], label='Training Mean MSE')
plt.plot(history.history['val_mean_loss_metric'], label='Validation Mean MSE')
plt.title('Mean MSE')
plt.xlabel('Epoch')
plt.ylabel('Mean MSE')
plt.legend()

# Plot std MSE
plt.subplot(2, 2, 3)
plt.plot(history.history['std_loss_metric'], label='Training Std MSE')
plt.plot(history.history['val_std_loss_metric'], label='Validation Std MSE')
plt.title('Std MSE')
plt.xlabel('Epoch')
plt.ylabel('Std MSE')
plt.legend()

# Plot weight CCE (categorical cross entropy)
plt.subplot(2, 2, 4)
plt.plot(history.history['weight_loss_metric'], label='Training Weight CCE')
plt.plot(history.history['val_weight_loss_metric'], label='Validation Weight CCE')
plt.title('Weight CCE')
plt.xlabel('Epoch')
plt.ylabel('Weight CCE')
plt.legend()

plt.tight_layout()
plt.savefig('./figures/training_validation_history.png')
plt.show()


# Function to plot Gaussian distributions
def plot_gaussians(means_true, stds_true, weights_true, means_pred, stds_pred, weights_pred, num_components):
    x = np.linspace(-10, 10, 1000)  

    # PDF computed for each component separately
    pdf_individual_true = [weights_true[i] * norm.pdf(x, means_true[i], stds_true[i]) for i in range(num_components)]
    pdf_individual_pred = [weights_pred[i] * norm.pdf(x, means_pred[i], stds_pred[i]) for i in range(num_components)]

    plt.figure(figsize=(8, 6))
    for i, pdf_i in enumerate(pdf_individual_true):
        plt.plot(x, pdf_i, alpha=0.7, label=f'True Gaussian {i+1}: mean={means_true[i]:.3f}, std={stds_true[i]:.3f}')
        plt.axvline(means_true[i], color='k', linestyle='dotted')
    for i, pdf_i in enumerate(pdf_individual_pred):
        plt.plot(x, pdf_i, alpha=0.7, linestyle='--', label=f'Pred Gaussian {i+1}: mean={means_pred[i]:.3f}, std={stds_pred[i]:.3f}')
        plt.axvline(means_pred[i], color='r', linestyle='dotted')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('True vs Predicted Gaussian Distributions')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/gaussian_comparison_val.png')
    plt.show()


# Get the model predictions 
# we should use validation data to evaluate performance of the model on useen data
#predictions = model.predict([train_forward_inputs, train_backward_inputs])
predictions = model.predict([val_forward_inputs, val_backward_inputs])


# Split the predictions into means, stds, and weights
means_pred = predictions[:, :num_components]
stds_pred = predictions[:, num_components:2*num_components]
weights_pred = predictions[:, 2*num_components:]

means_pred = means_pred.reshape(-1, num_components)
stds_pred = stds_pred.reshape(-1, num_components)
weights_pred = weights_pred.reshape(-1, num_components)

# extract means, stds, and weights from true values
means_true = val_y_true[:, :num_components]
stds_true = val_y_true[:, num_components:2*num_components]
weights_true = val_y_true[:, 2*num_components:]

# print predicted vs true values
print("means_true:", means_true)
print("means_pred:", means_pred)
print("stds_true:", stds_true)
print("stds_pred:", stds_pred)
print("weights_true:", weights_true)
print("weights_pred:", weights_pred)

# Reshape the predicted values 
means_pred = means_pred.reshape(-1, num_components)
stds_pred = stds_pred.reshape(-1, num_components)
weights_pred = weights_pred.reshape(-1, num_components)

# Plot the Gaussians
plot_gaussians(means_true[0], stds_true[0], weights_true[0], means_pred[0], stds_pred[0], weights_pred[0], num_components)