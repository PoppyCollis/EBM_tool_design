"""
Unconditioned prior over tool designs. Sampling from this prior will give you random tools.
Right now, it is independent factors of l1, l2, theta sampled from uniform distributions.
We will extend towards more expressive and complex priors in the future (e.g. diffusion prior).

Note that we return sin(theta) and cos(theta) so that the network understands that 0 is near 360
(i.e. we deal with the wrap-around problem).
"""

import numpy as np
import pandas as pd
from helpers.plots import visualise_tools
import torch
from torch.utils.data import Dataset
from config import ToolDatasetConfig


class ToolDataset:
    def __init__(self, l1_range, l2_range, theta_range, reward_type = "mse"):
        """
        l1_range/l2_range: tuples (min, max)
        theta_range: tuple (min_deg, max_deg)
        """
        self.l1_bounds = l1_range
        self.l2_bounds = l2_range
        self.theta_bounds = (np.radians(theta_range[0]), np.radians(theta_range[1]))
        self.reward_type = reward_type

    def sample_design(self, n_samples=1):
        """Samples tools using Uniform priors."""
        l1 = np.random.uniform(self.l1_bounds[0], self.l1_bounds[1], n_samples)
        l2 = np.random.uniform(self.l2_bounds[0], self.l2_bounds[1], n_samples)
        theta = np.random.uniform(self.theta_bounds[0], self.theta_bounds[1], n_samples)
        
        # 1. Pre-calculate sin/cos 
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        designs =  {
            'l1': l1,
            'l2': l2,
            'theta': theta,
            'sin_theta': sin_theta,
            'cos_theta': cos_theta
        }
        
        return designs
    
    def sample_target_location(self, n_samples, max_radius):
        # 1. Sample angles uniformly
        theta = np.random.uniform(0, 2 * np.pi, n_samples)
        
        # 2. Sample radius using the square root trick for uniformity
        # np.random.rand(n) gives values between [0, 1)
        r = max_radius * np.sqrt(np.random.rand(n_samples))
        
        # 3. Convert Polar coordinates to Cartesian (x, y)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.stack([x, y], axis=-1)
    
    def sample_dataset(self, n_samples=1):
        """Samples tools and random target location and calculates 
        reward as distance between the two to create a dummy dataset."""
        designs = self.sample_design(n_samples)
        
        max_radius = ((self.l1_bounds[1] - self.l1_bounds[0]) + (self.l2_bounds[1] - self.l2_bounds[0]))
        targets = self.sample_target_location(n_samples, max_radius)

        rewards = [] 

        # Calculate the end effector location with respect to the orgin of the tool at (0,0)
        for i in range(n_samples):
            l1 = designs['l1'][i]
            l2 = designs['l2'][i]
            theta = designs['theta'][i]
            
            # Calculate vertical orientation coordinates
            p1 = np.array([0, 0])           # Base
            p2 = np.array([0, l1])          # Joint (Vertical)
            p3 = np.array([l2 * np.sin(theta), l1 + l2 * np.cos(theta)]) # Tip
            
            # Reward shaping
            
            if self.reward_type == "euclidean_distance":
                reward = -np.sqrt( (p3[0]- targets[i][0])**2 + (p3[1] - targets[i][1])**2)
                
            elif self.reward_type == "mse":
                reward = -((p3[0]- targets[i][0])**2 + (p3[1] - targets[i][1])**2)
                
            elif self.reward_type == "saturated_euclidean_distance":
                reward = -np.tanh(np.sqrt( (p3[0]- targets[i][0])**2 + (p3[1] - targets[i][1])**2))
            else:
                raise NotImplementedError("Reward type not supported.")
            
            rewards.append(reward)
                
        rewards = np.array(rewards)
        
        dataset = designs.copy()
        dataset["end_effector_x"] = p3[0]
        dataset["end_effector_y"] = p3[1]
        dataset["x_target"] = targets[:,0]
        dataset["y_target"] = targets[:,1]
        dataset["reward"] =  rewards.flatten()
        return dataset
    
class CustomDataset(Dataset):
    def __init__(self, dataframe, dataset_stats=None, label_stats=None):
        # Load the data into memory once
        self.data = dataframe
        
        # Pre-convert columns to tensors to save time during training
        self.features = torch.tensor(self.data[['l1', 'l2', 'sin_theta', 'cos_theta', "x_target", "y_target"]].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['reward'].values, dtype=torch.float32).unsqueeze(1)
        
        # 2. Apply normalization if stats are provided
        if dataset_stats is not None:
            self.features = (self.features - dataset_stats['mean']) / dataset_stats['std']
            
        if label_stats is not None:
            self.labels = (self.labels - label_stats['mean']) / label_stats['std']
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def main():
    
    save_file = ToolDatasetConfig.SAVE_PATH
    
    # Get bounds from config file [Lower, Upper]
    l1_bounds = ToolDatasetConfig.L1_BOUNDS 
    l2_bounds = ToolDatasetConfig.L2_BOUNDS 
    theta_bounds = ToolDatasetConfig.THETA_BOUNDS   
    reward_type = ToolDatasetConfig.REWARD_TYPE
    
    
    tool_dataset = ToolDataset(l1_bounds, l2_bounds, theta_bounds, reward_type)

    num_designs = ToolDatasetConfig.NUM_DESIGNS
    
    # designs = tool_dataset.sample_design(num_designs)
    # visualise_tools(designs)
    # print(designs)
    data = tool_dataset.sample_dataset(num_designs)

    df = pd.DataFrame(data)
    df.to_parquet(save_file)
    
    

if __name__ == "__main__":
    main()    # need to define an arbitrary dummy reward to train the reward predictor with.
