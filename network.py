# neural network
# input: whole trajectory of the triple
# output: params of the gaussian fit to the distribution of lyapunov exponents
# using the distribution of slopes as ground truth

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
# 1) window size should be the optimal one found for the specific file - put it as one of the arguments
# positions_velocities should be of size (n_timesteps, 18) 
def preprocess_trajectory(file_path, gaussian_fit_params_path=None, n_steps=100):
    """
    Preprocess trajectory data for use in a neural network.
    
    Args:
        file_path (str): Path to the trajectory CSV file.
        gaussian_fit_params_path (str): Path to the precomputed Gaussian fit parameters file.
    
    Returns:
        forward_trajectories (dict): Dictionary of forward trajectories grouped by particle.
        backward_trajectories (dict): Dictionary of backward trajectories grouped by particle.
        gaussian_fit_params (np.ndarray): Gaussian fit parameters (ground truth).
    """
    # Load the trajectory data
    df = pd.read_csv(file_path, dtype=str)

    # Identify unique particles
    particles = df['Particle Number'].unique()

    # Identify position and velocity columns
    pos_vel_cols = ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']

    # Set precision and convert columns to high precision numbers
    set_precision(df, pos_vel_cols)
    for col in pos_vel_cols:
        df[col] = df[col].apply(string_to_mpf)

    # Separate forward and backward trajectories
    forward_trajectory = df[df['Phase'].astype(int) == 1]
    backward_trajectory = df[df['Phase'].astype(int) == -1]

    # load ground truth data
    gaussian_fit_params = np.loadtxt(gaussian_fit_params_path)
    
    # Group trajectories by particle and include just the first n_steps timesteps
    forward_trajectories = {particle: forward_trajectory[forward_trajectory['Particle Number'] == particle][pos_vel_cols].astype(float).to_numpy()[:n_steps] for particle in particles}
    backward_trajectories = {particle: backward_trajectory[backward_trajectory['Particle Number'] == particle][pos_vel_cols].astype(float).to_numpy()[:n_steps] for particle in particles}

 
    return forward_trajectories, backward_trajectories, gaussian_fit_params



def build_temporal_encoder(sequence_length, feature_dim, latent_dim):
    """
    Builds a temporal encoder for forward and backward trajectories.

    Args:
        sequence_length (int): Length of the input sequence (number of timesteps).
        feature_dim (int): Dimensionality of input features (e.g., 6 for positions and velocities).
        latent_dim (int): Dimension of the compact representation.

    Returns:
        model (tf.keras.Model): A Keras model encoding trajectory dynamics.
    """
    # Inputs for forward and backward trajectories
    forward_input = layers.Input(shape=(sequence_length, feature_dim), name='forward_trajectory_input')
    backward_input = layers.Input(shape=(sequence_length, feature_dim), name='backward_trajectory_input')

    # Split features into positions (first half) and velocities (second half)
    pos_dim = feature_dim // 2
    
    def split_pos_vel(x):
        positions = x[..., :pos_dim]
        velocities = x[..., pos_dim:]
        return positions, velocities

    # Process forward trajectory
    forward_pos, forward_vel = layers.Lambda(split_pos_vel)(forward_input)
    
    # Process backward trajectory (flip velocity signs)
    backward_pos, backward_vel = layers.Lambda(split_pos_vel)(backward_input)
    backward_vel_flipped = layers.Lambda(lambda x: -x)(backward_vel)

    # Recombine features
    forward_features = layers.Concatenate()([forward_pos, forward_vel])
    backward_features = layers.Concatenate()([backward_pos, backward_vel_flipped])

    # LSTM layers with attention to learn trajectory relationships
    forward_encoded = layers.LSTM(128, return_sequences=True, name='forward_lstm')(forward_features)
    backward_encoded = layers.LSTM(128, return_sequences=True, name='backward_lstm')(backward_features)

    # Self-attention to learn phase space relationships
    # to focus on specific parts of the input sequence
    attention = layers.Attention()([forward_encoded, backward_encoded])
    
    # Combine features with learned attention
    combined_features = layers.Concatenate(name='combined_features')([
        forward_encoded, 
        backward_encoded,
        attention
    ])

    # Flip the phase space distance values - this is for considering the right order
    flipped_combined_features = layers.Lambda(lambda x: tf.reverse(x, axis=[1]))(combined_features)

    # Aggregate with attention to phase space structure
    aggregated_features = layers.GlobalAveragePooling1D(name="aggregated_features")(flipped_combined_features)
    
    # Add extra layers to learn phase space structure
    x = layers.Dense(256, activation='relu')(aggregated_features)
    x = layers.Dense(128, activation='relu')(x)
    
    # Compact latent representation
    latent_representation = layers.Dense(latent_dim, activation='linear', name="latent_representation")(x)

    # Build the model
    model = models.Model(
        inputs=[forward_input, backward_input], 
        outputs=latent_representation, 
        name="trajectory_encoder"
    )
    return model

# qui toglierei num_components se le teniamo fisse
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

    # Heights (softplus activation for positivity)
    heights = layers.Dense(num_components, activation='softplus', name='gmm_heights')(x)

    # Build the model
    model = models.Model(inputs, [means, stds, weights, heights], name="gmm_decoder")
    return model

# to combine encoder and decoder 
def build_phase_space_model(sequence_length, feature_dim, latent_dim, num_components):
    """
    Builds a model to understand the phase space dynamics and predict GMM parameters.

    Args:
        sequence_length (int): Length of the input sequence (number of timesteps).
        feature_dim (int): Dimensionality of input features (e.g., 6 for positions and velocities).
        latent_dim (int): Dimension of the latent representation.
        num_components (int): Number of Gaussian components in the mixture.

    Returns:
        model (tf.keras.Model): A Keras model to process trajectories and output GMM parameters.
    """
    # Encoder
    encoder = build_temporal_encoder(sequence_length, feature_dim, latent_dim)

    # Decoder
    decoder = build_gmm_decoder(latent_dim, num_components)

    # Inputs for forward and backward trajectories
    forward_input = layers.Input(shape=(sequence_length, feature_dim), name='forward_trajectory_input')
    backward_input = layers.Input(shape=(sequence_length, feature_dim), name='backward_trajectory_input')

    # Pass through encoder
    latent_representation = encoder([forward_input, backward_input])

    # Pass latent representation through decoder
    means, stds, weights, heights = decoder(latent_representation)

    # Concatenate outputs into a single tensor
    outputs = layers.Concatenate(name="gmm_outputs")([means, stds, weights, heights])

    # Build the full model
    model = models.Model(inputs=[forward_input, backward_input], outputs=outputs, name="phase_space_model")
    return model









""" 
# giving the trajectories, slopes, psd as ground truth
def build_phase_space_model_with_trajectory_prediction(sequence_length, feature_dim, latent_dim, num_components, prediction_steps):
    
    Builds a model to process trajectories, predict their evolution, and output GMM parameters.
    
    Args:
        sequence_length (int): Length of the input sequence (number of timesteps).
        feature_dim (int): Dimensionality of input features (e.g., 6 for positions and velocities).
        latent_dim (int): Dimension of the latent representation.
        num_components (int): Number of Gaussian components in the mixture.
        prediction_steps (int): Number of timesteps to predict into the future.
    
    Returns:
        model (tf.keras.Model): A Keras model to process trajectories and output GMM parameters.
    
    # Encoder for trajectory dynamics
    encoder = build_temporal_encoder(sequence_length, feature_dim, latent_dim)

    # Decoder for Gaussian Mixture Model parameters
    gmm_decoder = build_gmm_decoder(latent_dim, num_components)

    # Inputs for forward and backward trajectories
    forward_input = layers.Input(shape=(sequence_length, feature_dim), name='forward_trajectory_input')
    backward_input = layers.Input(shape=(sequence_length, feature_dim), name='backward_trajectory_input')

    # Pass through encoder
    latent_representation = encoder([forward_input, backward_input])

    # GMM outputs (Lyapunov estimator components)
    means, stds, weights, heights = gmm_decoder(latent_representation)

    # Trajectory prediction head
    future_trajectory = layers.Dense(prediction_steps * feature_dim, activation='linear', name='trajectory_prediction')(latent_representation)
    future_trajectory = layers.Reshape((prediction_steps, feature_dim), name='reshaped_trajectory')(future_trajectory)

    # Phase space distance computation
    phase_space_distance = layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=-1)),
        name='phase_space_distance'
    )([forward_input, backward_input])

    # Slope distribution analysis head
    slope_features = layers.Dense(64, activation='relu', name='slope_dense_1')(phase_space_distance)
    slope_distribution = layers.Dense(32, activation='relu', name='slope_dense_2')(slope_features)
    slope_distribution = layers.Dense(num_components, activation='softmax', name='slope_distribution')(slope_distribution)

    # Combine all outputs
    outputs = {
        "gmm_outputs": layers.Concatenate(name="gmm_outputs")([means, stds, weights, heights]),
        "trajectory_prediction": future_trajectory,
        "phase_space_distance": phase_space_distance,
        "slope_distribution": slope_distribution,
    }

    # Build the full model
    model = models.Model(inputs=[forward_input, backward_input], outputs=outputs, name="phase_space_model_with_prediction")
    return model

    
# dont know if it's needed
def build_trajectory_aggregator(sequence_length, feature_dim, latent_dim, pooling_type="average"):

    Builds a model to aggregate trajectory features into a compact representation.

    Args:
        sequence_length (int): Length of the input sequence (number of timesteps).
        feature_dim (int): Dimensionality of input features (e.g., 6 for positions and velocities).
        latent_dim (int): Dimension of the compact representation.
        pooling_type (str): Type of pooling to use: 'average', 'max', or 'attention'.

    Returns:
        model (tf.keras.Model): A Keras model aggregating features into a compact representation.

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


 """













