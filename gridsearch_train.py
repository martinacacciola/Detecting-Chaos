import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import optimizers
from train_network_with_val import custom_gmm_loss, mean_loss_metric, std_loss_metric, weight_loss_metric
from sklearn.model_selection import GridSearchCV, train_test_split
from network import preprocess_trajectory, build_phase_space_model

# Define a function to create the model
""" def create_model(learning_rate=0.001, latent_dim=64, num_components=2, optimizer='adam', **kwargs):
    sequence_length = forward_inputs.shape[1]
    feature_dim = forward_inputs.shape[2]
    model = build_phase_space_model(sequence_length, feature_dim, latent_dim, num_components)([forward_inputs, backward_inputs])
    if optimizer == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=custom_gmm_loss,
        metrics=[mean_loss_metric, std_loss_metric, weight_loss_metric],
    )
    return model """
def create_model(learning_rate=0.001, latent_dim=64, num_components=2, optimizer='adam', **kwargs):
    # Placeholder dimensions
    sequence_length = forward_inputs.shape[1]
    feature_dim = forward_inputs.shape[2]
    
    # Define the single flattened input
    flat_input = tf.keras.layers.Input(shape=(sequence_length * feature_dim * 2,), name="flat_input")
    
    # Use fixed slicing for the input
    def split_inputs(x):
        forward = tf.reshape(x[:, :sequence_length * feature_dim], (-1, sequence_length, feature_dim))
        backward = tf.reshape(x[:, sequence_length * feature_dim:], (-1, sequence_length, feature_dim))
        return forward, backward
    
    # Split inputs explicitly using Lambda but define statically
    forward_input, backward_input = tf.keras.layers.Lambda(split_inputs, name="split_inputs")(flat_input)
    
    # Call your model-building function
    model_output = build_phase_space_model(sequence_length, feature_dim, latent_dim, num_components)(
        [forward_input, backward_input]
    )
    
    model = tf.keras.models.Model(inputs=flat_input, outputs=model_output)
    
    # Optimizer handling
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer '{optimizer}'. Must be one of ['adam', 'sgd', 'rmsprop'].")
    
    model.compile(
        optimizer=opt,
        loss=custom_gmm_loss,
        metrics=[mean_loss_metric, std_loss_metric, weight_loss_metric],
    )
    return model




# Load and preprocess the data
forward_trajectories, backward_trajectories, timesteps, deltas, window_slopes, gaussian_fit_params = preprocess_trajectory(
    file_path='./Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv',
    delta_per_step_path='./data/delta_per_step_L0_00_i1775_e90_Lw392.txt',
    window_slopes_path='./data/window_slopes_L0_00_i1775_e90_Lw392.txt',
    gaussian_fit_params_path='./data/gmm_parameters_L0_00_i1775_e90_Lw392.txt'
)

# Prepare the data
forward_inputs = np.array(list(forward_trajectories.values()))
backward_inputs = np.array(list(backward_trajectories.values()))
means_true = np.tile(gaussian_fit_params[0, :], (forward_inputs.shape[0], 1))
stds_true = np.tile(gaussian_fit_params[1, :], (forward_inputs.shape[0], 1))
weights_true = np.tile(gaussian_fit_params[2, :], (forward_inputs.shape[0], 1))
y_true = np.concatenate([means_true, stds_true, weights_true], axis=1)

# Split into training and testing sets
train_forward_inputs, val_forward_inputs, train_backward_inputs, val_backward_inputs, train_y_true, val_y_true = train_test_split(
    forward_inputs, backward_inputs, y_true, test_size=0.2, random_state=42
)

print("train_forward_inputs shape:", train_forward_inputs.shape)
print("train_backward_inputs shape:", train_backward_inputs.shape)
print("train_y_true shape:", train_y_true.shape)
print("val_forward_inputs shape:", val_forward_inputs.shape)
print("val_backward_inputs shape:", val_backward_inputs.shape)
print("val_y_true shape:", val_y_true.shape)

# Combine inputs for scikit-learn compatibility
train_inputs = [train_forward_inputs, train_backward_inputs]
val_inputs = [val_forward_inputs, val_backward_inputs]


# Flatten inputs for scikit-learn compatibility (GridSearchCV does not natively support multiple inputs)
train_inputs_flat = np.hstack((train_forward_inputs.reshape(train_forward_inputs.shape[0], -1),
                               train_backward_inputs.reshape(train_backward_inputs.shape[0], -1)))
val_inputs_flat = np.hstack((val_forward_inputs.reshape(val_forward_inputs.shape[0], -1),
                             val_backward_inputs.reshape(val_backward_inputs.shape[0], -1)))


# Print the shapes of the flattened inputs
print("train_inputs_flat shape:", train_inputs_flat.shape)
print("val_inputs_flat shape:", val_inputs_flat.shape)

# Define the model for scikit-learn
model = KerasRegressor(
    build_fn=create_model, # will be depracated, instead use model 
    verbose=0,
    learning_rate=0.001,  # Default value
    latent_dim=64,        # Default value
    num_components=2,     # Default value
    optimizer='adam'      # Default value
)

#model = KerasClassifier(build_fn=create_model, verbose=0)
# Print available parameters
print("Available parameters:", model.get_params().keys())

# Define the grid of hyperparameters to search
param_grid = {
    'learning_rate': [0.01, 0.1],
    'latent_dim': [128, 256],
    'batch_size': [64, 128],
    'epochs': [1], # set back to 100 if it works
    'optimizer': ['adam', 'sgd', 'rmsprop']
}

# Perform grid search
# to use cv=3 more samples in the training set are needed (also to use score r2)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring={
    'neg_mean_squared_error': 'neg_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error'}, refit='neg_mean_squared_error', verbose=1, n_jobs=-1 )
grid_result = grid.fit(train_inputs_flat, train_y_true)

# Print the best hyperparameters and their performance
print(f"Best parameters found: {grid_result.best_params_}")
print(f"Best score: {grid_result.best_score_}")

# Save the grid search results to a file
results_df = pd.DataFrame(grid_result.cv_results_)
results_df.to_csv('./saved_results/grid_search_results.csv', index=False)

# Evaluate the model on the validation data
best_model = grid_result.best_estimator_
val_loss = best_model.score(val_inputs_flat, val_y_true)
print(f"Validation Loss with Best Parameters: {val_loss}")
