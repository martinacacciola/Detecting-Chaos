import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from network import preprocess_trajectory, build_temporal_encoder, build_gmm_decoder, build_phase_space_model

def custom_gmm_loss(y_true, y_pred):
    """
    Custom GMM loss function for concatenated outputs.

    Args:
        y_true (tensor): True values, concatenated [means, stds, weights].
        y_pred (tensor): Predicted values, concatenated [means, stds, weights].

    Returns:
        Tensor: Combined loss value.
    """


    # Split y_true and y_pred into means, stds, and weights
    means_true, stds_true, weights_true = tf.split(y_true, num_or_size_splits=3, axis=1)
    means_pred, stds_pred, weights_pred = tf.split(y_pred, num_or_size_splits=3, axis=1)

    # Compute losses
    mean_loss = tf.reduce_mean(tf.square(means_true - means_pred))
    std_loss = tf.reduce_mean(tf.square(stds_true - stds_pred))
    weight_loss = tf.reduce_mean(losses.categorical_crossentropy(weights_true, weights_pred))

    return mean_loss + std_loss + weight_loss

# Custom metrics
def mean_loss_metric(y_true, y_pred):
    means_true, _, _ = tf.split(y_true, num_or_size_splits=3, axis=1)
    means_pred, _, _ = tf.split(y_pred, num_or_size_splits=3, axis=1)
    return tf.reduce_mean(tf.square(means_true - means_pred))

def std_loss_metric(y_true, y_pred):
    _, stds_true, _ = tf.split(y_true, num_or_size_splits=3, axis=1)
    _, stds_pred, _ = tf.split(y_pred, num_or_size_splits=3, axis=1)
    return tf.reduce_mean(tf.square(stds_true - stds_pred))

def weight_loss_metric(y_true, y_pred):
    _, _, weights_true = tf.split(y_true, num_or_size_splits=3, axis=1)
    _, _, weights_pred = tf.split(y_pred, num_or_size_splits=3, axis=1)
    return tf.reduce_mean(losses.categorical_crossentropy(weights_true, weights_pred))


# Load the data
forward_trajectories, backward_trajectories, timesteps, deltas, window_slopes, gaussian_fit_params = preprocess_trajectory(
    file_path='./Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv',
    delta_per_step_path='./data/delta_per_step_L0_00_i1775_e90_Lw392.txt',
    window_slopes_path='./data/window_slopes_L0_00_i1775_e90_Lw392.txt',
    gaussian_fit_params_path='./data/gmm_parameters_L0_00_i1775_e90_Lw392.txt'
)

# Inspect the shape of gaussian_fit_params
#print("Shape of gaussian_fit_params:", gaussian_fit_params.shape)

# Prepare the inputs and outputs
forward_inputs = np.array(list(forward_trajectories.values()))
backward_inputs = np.array(list(backward_trajectories.values()))

# Adjust based on the shape of gaussian_fit_params
""" means_true = gaussian_fit_params[0, :]
stds_true = gaussian_fit_params[1, :]
weights_true = gaussian_fit_params[2, :] """
# Adjust based on the shape of gaussian_fit_params
means_true = np.tile(gaussian_fit_params[0, :], (forward_inputs.shape[0], 1))
stds_true = np.tile(gaussian_fit_params[1, :], (forward_inputs.shape[0], 1))
weights_true = np.tile(gaussian_fit_params[2, :], (forward_inputs.shape[0], 1))

y_true = np.concatenate([means_true, stds_true, weights_true], axis=1)

# Split the data into training and validation sets
train_forward_inputs, val_forward_inputs, train_backward_inputs, val_backward_inputs, train_y_true, val_y_true = train_test_split(
    forward_inputs, backward_inputs, y_true, test_size=0.2, random_state=42
)

# Build the model
sequence_length = forward_inputs.shape[1]
feature_dim = forward_inputs.shape[2]
latent_dim = 128
num_components = 2 #means_true.shape[0]
#learning_rate = 0.01

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
    epochs=100,
    batch_size=32,
)

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

# Plot weight CCE
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
    x = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(12, 8))

    for i in range(num_components):
        # True Gaussian
        y_true = weights_true[i] * (1 / (stds_true[i] * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - means_true[i]) / stds_true[i]) ** 2
        )
        plt.plot(x, y_true, label=f"True Gaussian {i+1}", linestyle="dashed")

        # Predicted Gaussian
        y_pred = weights_pred[i] * (1 / (stds_pred[i] * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - means_pred[i]) / stds_pred[i]) ** 2
        )
        plt.plot(x, y_pred, label=f"Predicted Gaussian {i+1}")

    plt.title("True vs Predicted Gaussians")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig('./figures/gaussian_comparison_val.png')
    plt.show()


# Get the model predictions
predictions = model.predict([val_forward_inputs, val_backward_inputs])


# Split the predictions into means, stds, and weights
means_pred = predictions[:, :num_components]
stds_pred = predictions[:, num_components:2*num_components]
weights_pred = predictions[:, 2*num_components:]

means_pred = means_pred.reshape(-1, num_components)
stds_pred = stds_pred.reshape(-1, num_components)
weights_pred = weights_pred.reshape(-1, num_components)

# print predicted vs true values
print("means_true:", means_true)
print("means_pred:", means_pred)
print("stds_true:", stds_true)
print("stds_pred:", stds_pred)
print("weights_true:", weights_true)
print("weights_pred:", weights_pred)


# Reshape the predicted values if necessary
means_pred = means_pred.reshape(-1, num_components)
stds_pred = stds_pred.reshape(-1, num_components)
weights_pred = weights_pred.reshape(-1, num_components)

# Plot the Gaussians
plot_gaussians(means_true[0], stds_true[0], weights_true[0], means_pred[0], stds_pred[0], weights_pred[0], num_components)