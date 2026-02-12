"""
Neural network reward prediction model f(τ, c) → R; takes in tool design parameters τ 
and task description c and outputs a scalar reward
"""

class RewardModel:    
    def __init__(self, input_dim, hidden_dim=64):
        """
        Initializes the Reward Model.
        
        Args:
            input_dim (int): Dimension of the combined design and task vector.
            hidden_dim (int): Number of units in the hidden layers.
        """
        import torch
        import torch.nn as nn
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def predict(self, designs, task_description):
        """
        Predicts the reward for given designs and task.
        
        Args:
            designs (dict): Dictionary containing 'l1', 'l2', 'theta' arrays.
            task_description (np.ndarray): Context vector.
            
        Returns:
            np.ndarray: Predicted rewards.
        """
        import torch
        import numpy as np

        # Convert dictionary of arrays to a single feature matrix
        l1 = designs['l1']
        l2 = designs['l2']
        theta = designs['theta']
        
        # Stack features: [N, 3]
        tau = np.stack([l1, l2, theta], axis=1)
        
        # Tile task description to match number of samples: [N, task_dim]
        n_samples = tau.shape[0]
        c = np.tile(task_description, (n_samples, 1))
        
        # Concatenate design and task: [N, 3 + task_dim]
        x = np.concatenate([tau, c], axis=1)
        x_tensor = torch.FloatTensor(x)
        
        self.model.eval()
        with torch.no_grad():
            reward = self.model(x_tensor).numpy()
            
        return reward.flatten()
