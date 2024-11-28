import unittest
import numpy as np
from network import preprocess_trajectory

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
    
if __name__ == '__main__':
    unittest.main()