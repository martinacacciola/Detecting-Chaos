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
        positions_velocities, timesteps, deltas, window_slopes, gaussian_fit_params = preprocess_trajectory(
            self.file_path,
            window_size=40,
            delta_per_step_path=self.delta_per_step_path,
            window_slopes_path=self.window_slopes_path,
            gaussian_fit_params_path=self.gaussian_fit_params_path
        )
        
        # Check the output types
        self.assertIsInstance(positions_velocities, np.ndarray)
        self.assertIsInstance(timesteps, np.ndarray)
        self.assertIsInstance(deltas, np.ndarray)
        self.assertIsInstance(window_slopes, np.ndarray)
        
        # Check the output shapes
        print('positions_velocities:', positions_velocities[:5])
        print('shape of input:', positions_velocities.shape)
        self.assertEqual(positions_velocities.shape[1], 6)  # 6 for each particle, 3 particles
        #self.assertEqual(len(timesteps), positions_velocities.shape[0])
        #self.assertEqual(len(deltas), positions_velocities.shape[0])
    
if __name__ == '__main__':
    unittest.main()