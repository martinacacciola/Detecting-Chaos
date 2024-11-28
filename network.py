# neural network
# input: whole trajectory of the triple
# output: params of the gaussian fit to the distribution of lyapunov exponents
# using the distribution of slopes as ground truth

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
from mpmath import mp
import tensorflow as tf
from tensorflow.keras import layers, models

# Set global precision for high-precision arithmetic
def set_precision(df, columns):
    def count_decimals(value):
        if '.' in value:
            return len(value.split('.')[1])
        return 0

    max_decimal_places = max(
        df[col].apply(count_decimals).max() for col in columns
    )
    mp.dps = max_decimal_places
    mp.prec = 3.33 * max_decimal_places  # Precision to 3.33 times decimal places
    return max_decimal_places

# Function to convert string to mpmath floating point
def string_to_mpf(value):
    return mp.mpf(value)

# Preprocess function
# TODO: 
# 1) is it better to use directly window slopes as file? 
# 2) window size should be the optimal one found for the specific file
# 3) i think phases are not necessary as output
# positions_velocities should be of size (n_timesteps, 18) 
# should we distinguish between particles?
def preprocess_trajectory(file_path, window_size=40, delta_per_step_path=None, window_slopes_path=None, gaussian_fit_params_path=None):
    """
    Preprocess trajectory data for use in a neural network.
    
    Args:
        file_path (str): Path to the trajectory CSV file.
        delta_file_path (str): Path to the delta per step data file.
        window_size (int): Number of timesteps to include in each slope calculation window.
        delta_per_step_path (str): Path to the precomputed delta per step data file.
        window_slopes_path (str): Path to the precomputed window slopes data file.
        gaussian_fit_params_path (str): Path to the precomputed Gaussian fit parameters file.
    
    Returns:
        positions_velocities (np.ndarray): Array of positions and velocities.
        phases (np.ndarray): Phase labels for the trajectories.
        timesteps (np.ndarray): Normalized timesteps.
        deltas (np.ndarray): Logarithm of delta values.
        T_norm_midpoints (list): Midpoints of normalized timesteps for slope calculation.
        window_slopes (np.ndarray): Windowed slopes of phase-space distances (ground truth).
    """
    # Load the trajectory data
    df = pd.read_csv(file_path, dtype=str)
    #pos_vel_cols = {}
    particles = df['Particle Number'].unique()
    # Identify position and velocity columns for each particle
    #for particle in particles:
    pos_vel_cols = ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']

    # Set precision and convert columns to high precision numbers
    set_precision(df, pos_vel_cols)
    for col in pos_vel_cols:
        df[col] = df[col].apply(string_to_mpf)

    # Get unique timesteps
    timesteps = df['Timestep'].unique()
    #total_timesteps = len(timesteps)

    # Separate forward and backward trajectories
    forward_trajectory = df[df['Phase'].astype(int) == 1]
    backward_trajectory = df[df['Phase'].astype(int) == -1]

    # Normalize lifetime using crossing time
    T_c = mp.mpf(2) * mp.sqrt(2)
    T_norm = [mp.mpf(timestep) / T_c for timestep in timesteps]

    if delta_per_step_path and window_slopes_path and gaussian_fit_params_path:
    # just considering case in which we provide directly the files
        delta_per_step = np.loadtxt(delta_per_step_path)
        window_slopes = np.loadtxt(window_slopes_path)
        gaussian_fit_params = np.loadtxt(gaussian_fit_params_path)
    else:
        raise ValueError("Paths for delta_per_step, window_slopes, and gaussian_fit_params must be provided.")
    #else:
        #delta_per_step = np.loadtxt(delta_file_path)
        #window_slopes = np.zeros(len(delta_per_step))  # Placeholder, replace with actual computation if needed
        #gaussian_fit_params = np.zeros(len(delta_per_step))  # Placeholder, replace with actual computation if needed

    
    # Convert processed arrays into NumPy arrays for neural network use
    positions_velocities = df[pos_vel_cols].astype(float).to_numpy()
    #phases = df['Phase'].astype(int).to_numpy()
    timesteps = np.array(T_norm, dtype=float)
    deltas = np.array(np.log(delta_per_step), dtype=float)

    return positions_velocities, timesteps, deltas, window_slopes, gaussian_fit_params



def build_temporal_encoder(input_shape, latent_dim):
    """
    Builds a temporal encoder for trajectory data using LSTM layers.

    Args:
        input_shape (tuple): Shape of the input data (sequence_length, feature_dim).
        - sequence_length (int): Length of the input sequence (number of timesteps).
        - feature_dim (int): Dimensionality of input features (e.g., 18 for positions and velocities for three bodies).
        latent_dim (int): Dimension of the latent space for encoded dynamics.

    Returns:
        model (tf.keras.Model): A Keras model encoding dynamics into latent representation.
    """
    # should we set sequence length = number of timesteps?
    inputs = layers.Input(shape=input_shape, name='trajectory_input')

    # LSTM layers to process temporal information
    # Return sequences for stacking multiple layers

    # processes input seq and outputs seq of hidden states
    x = layers.LSTM(128, return_sequences=True, activation='tanh', name='lstm_1')(inputs)
    # summarizes the entire seq into single hidden state vector
    x = layers.LSTM(128, return_sequences=False, activation='tanh', name='lstm_2')(x)  # Final representation

    # Latent representation layer (to encode dynamics of the traj)
    # dense layer maps the final hidden state to the desired latent dimension
    latent = layers.Dense(latent_dim, activation='linear', name='latent_layer')(x)

    # Build model
    model = models.Model(inputs, latent, name='temporal_encoder')
    return model



def build_trajectory_aggregator(sequence_length, feature_dim, latent_dim, pooling_type="average"):
    """
    Builds a model to aggregate trajectory features into a compact representation.

    Args:
        sequence_length (int): Length of the input sequence (number of timesteps).
        feature_dim (int): Dimensionality of input features (e.g., 6 for positions and velocities).
        latent_dim (int): Dimension of the compact representation.
        pooling_type (str): Type of pooling to use: 'average', 'max', or 'attention'.

    Returns:
        model (tf.keras.Model): A Keras model aggregating features into a compact representation.
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name='trajectory_input')

    # Temporal feature processing with LSTM
    x = layers.LSTM(128, return_sequences=True, activation='tanh', name='lstm')(inputs)

    # Apply pooling to aggregate features
    if pooling_type == "average":
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    elif pooling_type == "max":
        x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    elif pooling_type == "attention":
        # Attention pooling layer
        query = layers.Dense(128, activation='tanh', name="attention_query")(x)
        attention_scores = layers.Dense(1, activation='softmax', name="attention_scores")(query)
        x = tf.reduce_sum(attention_scores * x, axis=1)  # Weighted sum based on attention scores
    else:
        raise ValueError("Invalid pooling_type. Choose from 'average', 'max', or 'attention'.")

    # Compact representation layer
    compact_representation = layers.Dense(latent_dim, activation='linear', name="compact_representation")(x)

    # Build the model
    model = models.Model(inputs, compact_representation, name="trajectory_aggregator")
    return model


def build_gmm_decoder(latent_dim, num_components):
    """
    Builds a neural network decoder to map latent representations to Gaussian Mixture Model (GMM) parameters.

    Args:
        latent_dim (int): Dimension of the latent input representation.
        num_components (int): Number of Gaussian components in the mixture.

    Returns:
        model (tf.keras.Model): A Keras model that decodes latent representations to GMM parameters.
    """
    inputs = layers.Input(shape=(latent_dim,), name='latent_input')

    # Fully connected hidden layers for feature transformation
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)

    # Output layers for GMM parameters
    # Mean (linear activation)
    means = layers.Dense(num_components, activation='linear', name='gmm_means')(x)

    # Standard deviation (softplus activation for positivity)
    stds = layers.Dense(num_components, activation='softplus', name='gmm_stds')(x)

    # Weights (softmax activation to ensure they sum to 1)
    weights = layers.Dense(num_components, activation='softmax', name='gmm_weights')(x)

    # Build the model
    model = models.Model(inputs, [means, stds, weights], name="gmm_decoder")
    return model




""" class TrajectoryToGMM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2):
        super(TrajectoryToGMM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc_means = nn.Linear(hidden_dim * 2, 2)  # 2 means
        self.fc_stds = nn.Linear(hidden_dim * 2, 2)   # 2 std deviations
        self.fc_weights = nn.Linear(hidden_dim * 2, 2)  # 2 weights

    def forward(self, x):
        # LSTM for temporal feature extraction
        _, (hidden, _) = self.lstm(x)  # hidden: (num_layers * 2, batch_size, hidden_dim)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate forward and backward hidden states
        
        # Decode to GMM parameters
        means = self.fc_means(hidden)
        stds = F.softplus(self.fc_stds(hidden))  # Ensure positivity
        weights = F.softmax(self.fc_weights(hidden), dim=1)  # Ensure sum to 1
        
        return weights, means, stds
 """


"""  
this in preprocess_trajectory if we want to compute ground truth in this file

   else:
        # Load delta_per_step data
        delta_per_step = np.loadtxt(delta_file_path)
        
        # Flip delta_per_step for consistent slope calculation
        delta_flip = np.flip(delta_per_step)

        # Logarithm of delta
        delta_log = np.log10(np.array(delta_flip, dtype=float))

        # Compute midpoints for T_norm intervals
        T_norm_midpoints = [(T_norm[i] + T_norm[i + 1]) / 2 for i in range(len(T_norm) - 1)]

        # Compute window slopes for ground truth
        window_slopes = []
        window_midpoints = []

        for start_idx in range(0, len(delta_flip) - window_size + 1):
            end_idx = start_idx + window_size

            # Select data within the window
            delta_window = delta_flip[start_idx:end_idx]
            T_norm_window = T_norm[start_idx:end_idx]

            delta_log_window = np.log10(np.array(delta_window, dtype=float))
            
            # Compute slope over the window
            # these values represent the ground truth
            slope = (delta_log_window[-1] - delta_log_window[0]) / (T_norm_window[-1] - T_norm_window[0])
            window_slopes.append(float(slope))

            # Record the midpoint of the current time window
            window_midpoints.append((T_norm_window[0] + T_norm_window[-1]) / 2)

        # Convert window slopes to NumPy array
        window_slopes = np.array(window_slopes, dtype=float)

    # Convert processed arrays into NumPy arrays for neural network use
    positions_velocities = df[pos_vel_cols].astype(float).to_numpy()
    phases = df['Phase'].astype(int).to_numpy()
    timesteps = np.array(T_norm, dtype=float)
    deltas = np.array(delta_log, dtype=float) """