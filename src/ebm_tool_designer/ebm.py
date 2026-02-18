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
        self.reward_model, self.feature_stats, self.label_stats = self.load_reward_model(weights_path) # load in pretrained weights
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
        # Move stats dictionaries to device
        feature_stats = {k: v.to(self.device) for k, v in checkpoint['feature_stats'].items()}
        label_stats = {k: v.to(self.device) for k, v in checkpoint['label_stats'].items()}
        # also need to load in normalisation stats for features
        reward_model.eval()
        reward_model.to(self.device)
        return reward_model, feature_stats, label_stats
    
    def joint_energy(self, tau, c_target, r_target):
        # slice tau to get component tool parameters
        l1 = tau[:, 0:1]
        l2 = tau[:, 1:2]
        theta = tau[:, 2:3]
        
        si = torch.sin(theta) # need these for reward prediction network
        co = torch.cos(theta)
        
        # concatenate into a single input vector for reward prediction network
        x = torch.cat([l1, l2, si, co, c_target], dim=-1)
        
        # apply the same normalization used during training
        x = (x - self.feature_stats['mean']) / self.feature_stats['std']
        
        # also normalise the reward
        r_target = (r_target - self.label_stats['mean']) / self.label_stats['std']

        cond_energy = self.reward_model.energy(x, r_target)
        
        prior_energy = 0 # placeholder
        
        return cond_energy + prior_energy
    
    def langevin_dynamics(self, c_target, r_target, batch_size=32):
        
        # NEED TO ADD NOISE KICK TO LANGEVIN, AND MAYBE METROPOLIS-HASTINGS CORRECTED SAMPLING?
        
        # 1. start with a sample in tau space
        tau = self.prior.sample(batch_size)
        print(" initial tau:", tau.cpu().detach().numpy())

        # 2. translate to phi space
        # 2. translate to phi space
        phi = self.prior.transform_to_phi(tau).detach().requires_grad_(True) # now we are tracking grads, phi is the leaf node
        
        c_target = c_target.detach()
        r_target = r_target.detach()

        # 3. optimise in phi space 
        #optimizer = torch.optim.Adam([phi], lr=self.eta)
        
        energy_hist = []
        for i in range(self.n_sampling_steps):
            
            if phi.grad is not None:
                phi.grad.zero_()
                
            tau_current = prior.transform_to_tau(phi)
            
            sigmoid_phi = torch.sigmoid(phi)
            log_det_jacobian = torch.log((self.prior.bounds_high - self.prior.bounds_low) * sigmoid_phi * (1 - sigmoid_phi) + 1e-8).sum()
            energy = self.joint_energy(tau_current, c_target, r_target) - log_det_jacobian
                
            energy.backward()
            
            with torch.no_grad():
                energy_hist.append(energy.item())
                
                noise = torch.randn_like(phi) * torch.sqrt(torch.tensor(2.0 * self.eta))
                phi -= self.eta * phi.grad + noise
            
        tau_final = tau_current
        
        print("final tau:", tau_final.cpu().detach().numpy())
        
        return tau_final, energy_hist
        
device = EBMConfig.DEVICE

n_samples = 1

prior = ToolDesignPrior(ToolDatasetConfig.L1_BOUNDS, ToolDatasetConfig.L2_BOUNDS, ToolDatasetConfig.THETA_BOUNDS, ToolDatasetConfig.DEVICE)
        
ebm = EnergyBasedModel(prior, weights_path=RewardModelConfig.WEIGHTS_SAVE_PATH)


# sample a random target location and set a reward target

max_radius = prior.bounds_high[0] + prior.bounds_high[1]
theta = torch.rand(n_samples, device=device) * 2 * np.pi
r = max_radius * torch.sqrt(torch.rand(n_samples, device=device))
x = r * torch.cos(theta)
y = r * torch.sin(theta)
# Combine into a single tensor of shape (n_samples, 2)
c_target = torch.stack([x, y], dim=-1)

r_target = torch.tensor([-50.0], device=device) # whats an appropriate reward?

print(f"Target location: {c_target.cpu().detach().numpy()}, Reward target: {r_target.item()}")

tool_sample, energy_hist = ebm.langevin_dynamics(c_target, r_target, batch_size=1)


import matplotlib.pyplot as plt

plt.plot(energy_hist)
plt.show()



# todo:
# add log det of jacobian to overleaf
# add MH correction?
# optimise lr and iter size, check energy is decreasing and converging
# check predicted reward of final tool is close to reward 

# plot the tool sample and visualise w.r.t. target location and show reward!
