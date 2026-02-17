"""
Convert tool design prior and reward prediction model into an energy-based model which we can sample from.
"""
import numpy as np
from reward_model import MLP
import torch
from tool_design_prior import ToolDesignPrior
from config import ToolDatasetConfig, EBMConfig, RewardModelConfig
    
class EnergyBasedModel:
    def __init__(self, prior, weights_path):
        """
        Initializes the EBM with a prior and a reward model.
        
        Args:
            prior (ToolDesignPrior): The unconditioned prior over tool designs.
            weights_path (str): Path to the pre-trained reward model weights.
        """
        self.device = RewardModelConfig.DEVICE
        self.prior = prior
        self.reward_model = self.load_reward_model(weights_path) # load in pretrained weights
        self.n_sampling_steps = EBMConfig.N_SAMPLING_STEPS
        self.eta = EBMConfig.ETA
        
    def load_reward_model(self, weights_path):
        # load in architecture (must match one used for pre-trained model weights)
        reward_model = MLP(
            in_features=RewardModelConfig.IN_FEATURES, 
            hidden_features=RewardModelConfig.HIDDEN_FEATURES, 
            out_features=RewardModelConfig.OUT_FEATURES) 
        # load the pre-trained weights into the model
        checkpoint = torch.load(weights_path)
        reward_model.load_state_dict(checkpoint['model_state_dict'])
        reward_model.eval()
        reward_model.to(self.device)
        return reward_model
    
    def joint_energy(self, l1, l2, theta, c_target, r_target):
        # current sample, \tau, c, reward target
        # conditional energy + prior energy [batch_size, 6]
        si = torch.sin(theta)
        co = torch.cos(theta)
        
        x = torch.cat([l1, l2, si, co, c_target], dim=-1)
        
        cond_energy = self.reward_model.energy(x, r_target)
        
        prior_energy = 0.0
        
        return cond_energy + prior_energy

    def langevin_dynamics(self, c_target, r_target, batch_size=32):
        # 1. Initialize tool params, tau, from the prior (e.g., [batch_size, 4])
        # Assume tau is [l1, l2, sin, cos] 
        # also should only optimise theta not sin(theta)+cos(theta)
        
        # Initial samples from your Prior (l1, l2, theta only)
        l1_init,l2_init,theta_init = self.prior.sample(batch_size)
        
        tau = torch.cat([torch.tensor(l1_init).unsqueeze(1), 
                     torch.tensor(l2_init).unsqueeze(1),
                     torch.tensor(theta_init).unsqueeze(1)], dim=-1).to(self.device)
        tau.requires_grad_(True)
        
        
        c_target = c_target.to(self.device).detach()

        for t in range(self.n_sampling_steps):
            # 2. Slice tau to get individual components for
            # the Energy logic (which works on theta not sin(theta))
            l1 = tau[:, 0:1]
            l2 = tau[:, 1:2]
            theta = tau[:, 2:3]
        
            E = self.joint_energy(l1, l2, theta, c_target, r_target)
            E.sum().backward()
            
            with torch.no_grad():
                epsilon = torch.randn_like(tau) * np.sqrt(self.eta) # noise term
                tau -= (self.eta / 2) * tau.grad + epsilon
                
                # Projection/Clipping: Keep l1 and l2 within physical bounds?
            
            tau.grad.zero_()
            
        return tau.detach()

prior = ToolDesignPrior(ToolDatasetConfig.L1_BOUNDS, ToolDatasetConfig.L2_BOUNDS, ToolDatasetConfig.THETA_BOUNDS)
        
ebm = EnergyBasedModel(prior, weights_path=RewardModelConfig.WEIGHTS_SAVE_PATH)
    
print(ebm)


# CONSIDERATIONS:

# what doe the log uniform prior look like in the joint energy, it is not gaussian

# is langevin noise always gaussian or does it depend on the prior?

# The $\sin(\theta)^2 + \cos(\theta)^2 = 1$ Constraint. sample 3: $(l_1, l_2, \theta)$, This guarantees the angle is always valid. But does it struggle with the wrap around problem

# Normalization Alignment: the reward model was trained on normalised data: apply feature_stats to tau and c inside the joint energy function