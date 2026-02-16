"""
Convert tool design prior and reward prediction model into an energy-based model which we can sample from.
"""
import numpy as np


class ToolDesignPrior:
    def __init__(self, l1_bounds, l2_bounds, theta_bounds):
        self.l1_bounds = l1_bounds
        self.l2_bounds = l2_bounds
        self.theta_bounds = theta_bounds
        
        
    def sample(self, n_samples=1):
        
        l1 = np.random.uniform(self.l1_bounds[0], self.l1_bounds[1], n_samples)
        l2 = np.random.uniform(self.l2_bounds[0], self.l2_bounds[1], n_samples)
        theta = np.random.uniform(self.theta_bounds[0], self.theta_bounds[1], n_samples)
        
        # 1. Pre-calculate sin/cos 
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        return l1,l2,sin_theta,cos_theta
    
class EnergyBasedModel:    
    def __init__(self, prior, reward_model):
        """
        Initializes the EBM with a prior and a reward model.
        
        Args:
            prior (ToolDesignPrior): The unconditioned prior over tool designs.
            reward_model (RewardModel): The neural network predicting reward.
        """
        self.prior = prior
        self.reward_model = reward_model
        #load in reward model weights

    def energy(self):
        """
        Calculates the energy for a given set of designs and task.
        """        
        # Get reward prediction
        E = ...
        # Energy is negative log-probability
        return E

    def sample(self, task_description, n_samples=1, method='langevin'):
        """
        Samples from the EBM distribution p(τ | c) ∝ exp(-E(τ, c)).
        """
        if method == 'langevin':
            pass
        if method == "MCMC":
            pass
        else:
            raise NotImplementedError("Sampling method not supported.")




    # --- LOADING ---
    # 1. Re-create the model architecture first
    model = MLP(in_features=6)

    # 2. Load the dictionary
    checkpoint = torch.load(save_path)

    # 3. Load the weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # 4. Don't forget to set to eval mode if you're done training
    model.eval()
