import unittest
import numpy as np
from network import preprocess_trajectory, build_temporal_encoder, build_gmm_decoder, build_phase_space_model
import tensorflow as tf
from tensorflow.keras import layers, models

class TestPreprocessTrajectory(unittest.TestCase):

    def setUp(self):
        # Use the known paths for the data files
        self.file_path = './Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv'
        self.delta_per_step_path = './data/delta_per_step_L0_00_i1775_e90_Lw392.txt'
        self.window_slopes_path = './data/window_slopes_L0_00_i1775_e90_Lw392.txt'
        self.gaussian_fit_params_path = './data/gmm_parameters_L0_00_i1775_e90_Lw392.txt'

    def test_preprocess_trajectory(self):
        forward_trajectories, backward_trajectories, timesteps, deltas, window_slopes, gaussian_fit_params = preprocess_trajectory(
            self.file_path,
            window_size=40,
            delta_per_step_path=self.delta_per_step_path,
            window_slopes_path=self.window_slopes_path,
            gaussian_fit_params_path=self.gaussian_fit_params_path
        )
        
        # Check the output types
        self.assertIsInstance(forward_trajectories, dict)
        self.assertIsInstance(backward_trajectories, dict)
        self.assertIsInstance(timesteps, np.ndarray)
        self.assertIsInstance(deltas, np.ndarray)
        self.assertIsInstance(window_slopes, np.ndarray)
        self.assertIsInstance(gaussian_fit_params, np.ndarray)
        
        # Check the output shapes and contents
        for particle in forward_trajectories:
            self.assertIsInstance(forward_trajectories[particle], np.ndarray)
            self.assertIsInstance(backward_trajectories[particle], np.ndarray)
            self.assertEqual(forward_trajectories[particle].shape[1], 6)  # 6 for each particle, 3 particles
            self.assertEqual(backward_trajectories[particle].shape[1], 6)  # 6 for each particle, 3 particles

       # self.assertEqual(len(timesteps), len(deltas))
       # self.assertEqual(len(timesteps), len(window_slopes))
       # self.assertEqual(len(timesteps), len(gaussian_fit_params))
    
class TestBuildTemporalEncoder(unittest.TestCase):

    def setUp(self):
        # Use the known paths for the data files
        self.file_path = './Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv'
        self.delta_per_step_path = './data/delta_per_step_L0_00_i1775_e90_Lw392.txt'
        self.window_slopes_path = './data/window_slopes_L0_00_i1775_e90_Lw392.txt'
        self.gaussian_fit_params_path = './data/gmm_parameters_L0_00_i1775_e90_Lw392.txt'

    def test_build_temporal_encoder(self):
        # Preprocess the trajectory data to get real input
        forward_trajectories, backward_trajectories, timesteps, deltas, window_slopes, gaussian_fit_params = preprocess_trajectory(
            self.file_path,
            window_size=40,
            delta_per_step_path=self.delta_per_step_path,
            window_slopes_path=self.window_slopes_path,
            gaussian_fit_params_path=self.gaussian_fit_params_path
        )

        # Use one of the forward and backward trajectories as input
        sample_forward_input = list(forward_trajectories.values())[0]
        sample_backward_input = list(backward_trajectories.values())[0]

        # Define input shape and latent dimension
        sequence_length = sample_forward_input.shape[0]  # Number of timesteps
        feature_dim = sample_forward_input.shape[1]  # Feature dimension (e.g., 6 features for each particle)
        latent_dim = 64  # Example latent dimension

        #input_shape = (sequence_length, feature_dim)

        # Build the model
        model = build_temporal_encoder(sequence_length, feature_dim, latent_dim)

        # Check if the model is an instance of tf.keras.Model
        self.assertIsInstance(model, tf.keras.Model)

        # Check the model input shape
        self.assertEqual(model.input_shape, [(None, sequence_length, feature_dim), (None, sequence_length, feature_dim)])

        # Check the model output shape
        self.assertEqual(model.output_shape, (None, latent_dim))

        # Get the model output
        sample_output = model([np.expand_dims(sample_forward_input, axis=0), np.expand_dims(sample_backward_input, axis=0)])

        # Check the output shape
        self.assertEqual(sample_output.shape, (1, latent_dim))

class TestBuildGMMDecoder(unittest.TestCase):

    def setUp(self):
        # Use the known paths for the data files
        self.file_path = './Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv'
        self.delta_per_step_path = './data/delta_per_step_L0_00_i1775_e90_Lw392.txt'
        self.window_slopes_path = './data/window_slopes_L0_00_i1775_e90_Lw392.txt'
        self.gaussian_fit_params_path = './data/gmm_parameters_L0_00_i1775_e90_Lw392.txt'

    def test_build_gmm_decoder(self):
        # Preprocess the trajectory data to get real input
        forward_trajectories, backward_trajectories, timesteps, deltas, window_slopes, gaussian_fit_params = preprocess_trajectory(
            self.file_path,
            window_size=40,
            delta_per_step_path=self.delta_per_step_path,
            window_slopes_path=self.window_slopes_path,
            gaussian_fit_params_path=self.gaussian_fit_params_path
        )

        # Use one of the latent representations as input
        sample_forward_input = list(forward_trajectories.values())[0]
        sample_backward_input = list(backward_trajectories.values())[0]

        # Define input shape and latent dimension
        sequence_length = sample_forward_input.shape[0]  # Number of timesteps
        feature_dim = sample_forward_input.shape[1]  # Feature dimension (e.g., 6 features for each particle)
        latent_dim = 64  # Example latent dimension
        num_components = 5  # Example number of Gaussian components

        # Build the temporal encoder model
        encoder = build_temporal_encoder(sequence_length, feature_dim, latent_dim)
        # Get the latent representation
        latent_representation = encoder([np.expand_dims(sample_forward_input, axis=0), np.expand_dims(sample_backward_input, axis=0)])

        # Build the GMM decoder model
        decoder = build_gmm_decoder(latent_dim, num_components)

        # Check if the model is an instance of tf.keras.Model
        self.assertIsInstance(decoder, tf.keras.Model)

        # Check the model input shape
        self.assertEqual(decoder.input_shape, (None, latent_dim))

        # Check the model output shapes
        self.assertEqual(decoder.output_shape, [(None, num_components), (None, num_components), (None, num_components)])

        # Get the model output
        means, stds, weights = decoder(latent_representation)

        # Check the output shapes
        self.assertEqual(means.shape, (1, num_components))
        self.assertEqual(stds.shape, (1, num_components))
        self.assertEqual(weights.shape, (1, num_components))



class TestBuildPhaseSpaceModel(unittest.TestCase):

    def setUp(self):
        # Use the known paths for the data files
        self.file_path = './Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv'
        self.delta_per_step_path = './data/delta_per_step_L0_00_i1775_e90_Lw392.txt'
        self.window_slopes_path = './data/window_slopes_L0_00_i1775_e90_Lw392.txt'
        self.gaussian_fit_params_path = './data/gmm_parameters_L0_00_i1775_e90_Lw392.txt'

    def test_build_phase_space_model(self):
        # Preprocess the trajectory data to get real input
        forward_trajectories, backward_trajectories, timesteps, deltas, window_slopes, gaussian_fit_params = preprocess_trajectory(
            self.file_path,
            window_size=40,
            delta_per_step_path=self.delta_per_step_path,
            window_slopes_path=self.window_slopes_path,
            gaussian_fit_params_path=self.gaussian_fit_params_path
        )

        # Use one of the forward and backward trajectories as input
        sample_forward_input = list(forward_trajectories.values())[0]
        sample_backward_input = list(backward_trajectories.values())[0]

        # Define input shape and latent dimension
        sequence_length = sample_forward_input.shape[0]  # Number of timesteps
        feature_dim = sample_forward_input.shape[1]  # Feature dimension (e.g., 6 features for each particle)
        latent_dim = 64  # Example latent dimension
        num_components = 5  # Example number of Gaussian components

         # Build the phase space model
        model = build_phase_space_model(sequence_length, feature_dim, latent_dim, num_components)

        # Check if the model is an instance of tf.keras.Model
        self.assertIsInstance(model, tf.keras.Model)

        # Check the model input shape
        self.assertEqual(model.input_shape, [(None, sequence_length, feature_dim), (None, sequence_length, feature_dim)])

        # Check the model output shapes
        self.assertEqual(model.output_shape, [(None, num_components), (None, num_components), (None, num_components)])

        # Get the model output
        means, stds, weights = model([np.expand_dims(sample_forward_input, axis=0), np.expand_dims(sample_backward_input, axis=0)])

        # Check the output shapes
        self.assertEqual(means.shape, (1, num_components))
        self.assertEqual(stds.shape, (1, num_components))
        self.assertEqual(weights.shape, (1, num_components))

if __name__ == '__main__':
    unittest.main()