"""
Convert tool design prior and reward prediction model into an energy-based model which we can sample from.
"""
import numpy as np
from reward_model import MLP
import torch
from tool_design_prior import ToolDesignPrior
from tool_dataset import ToolDataset
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
        feature_stats = checkpoint['feature_stats']
        label_stats = checkpoint['label_stats']
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
        
        # 1. start with a sample in tau space
        tau = self.prior.sample(batch_size)
        print(" initial tau:", tau)

        # 2. translate to phi space
        # 2. translate to phi space
        phi = self.prior.transform_to_phi(tau) # now we are tracking grads, phi is the leaf node
        print(phi)

        # 3. optimise in phi space
        optimizer = torch.optim.Adam([phi], lr=self.eta)
        
        for i in range(self.n_sampling_steps):
            
            optimizer.zero_grad() 
        
            tau_current = prior.transform_to_tau(phi) # re-derive tau from phi inside the loop.

            energy = self.joint_energy(self, tau_current, c_target, r_target)
        
            # backprop
            energy.backward()
        
            optimizer.step()
            
        tau_final = tau_current
        
        return tau_final
        
    
n_samples = 1

prior = ToolDesignPrior(ToolDatasetConfig.L1_BOUNDS, ToolDatasetConfig.L2_BOUNDS, ToolDatasetConfig.THETA_BOUNDS, ToolDatasetConfig.DEVICE)
        
ebm = EnergyBasedModel(prior, weights_path=RewardModelConfig.WEIGHTS_SAVE_PATH)


# sample a random target location

max_radius = prior.bounds_high[0] + prior.bounds_high[1]
theta = torch.rand(n_samples) * 2 * np.pi
r = max_radius * torch.sqrt(torch.rand(n_samples))
x = r * torch.cos(theta)
y = r * torch.sin(theta)
# Combine into a single tensor of shape (n_samples, 2)
c_target = torch.stack([x, y], dim=-1)

r_target = torch.tensor([-50.0]) # whats an appropriate reward?

# tool_sample = ebm.langevin_dynamics(c_target, r_target, batch_size=1)




# tests: 
# 1. get two tool vectors and work out the joint energy of each
# 2. check energy is higher when reward pred is further from target
# 3. 


# CONSIDERATIONS:

# The $\sin(\theta)^2 + \cos(\theta)^2 = 1$ Constraint. sample 3: $(l_1, l_2, \theta)$, This guarantees the angle is always valid.

# the reward model was trained on normalised data: apply feature_stats to tau and c inside the joint energy function - 
# If your reward model was trained on normalized data, you should be performing Langevin in the normalized space.

# MALA combines Langevin Dynamics with a Metropolis-Hastings acceptance step. 


"""
        
        # 1. INITIALISE tool params, tau, by sampling from the prior (e.g., [batch_size, 4])
        
        l1_init,l2_init,theta_init = self.prior.sample(batch_size)
        
        tau = torch.stack([l1_init, l2_init, theta_init], dim=-1).to(self.device).requires_grad_(True)
        
        c_target = c_target.to(self.device).detach()
        
        # 2. TRANSFORM tau by converting into unconstrained space (phi) using sigmoid
        
        phi = self.transform_to_phi(tau)

        for i in range(self.n_sampling_steps):
            
            # 3. ENERGY CALCULATION
        
            # Convert phi back to physical tau inside the gradient tape
            current_tau, log_jacob = self.transform_to_physical_space(phi)
            
            # Calculate energy using your existing joint_energy logic
            # slice tau to get individual components for the Energy logic (which works on theta not sin(theta))
            l1_current = current_tau[:, 0:1]
            l2_current = current_tau[:, 1:2]
            theta_current = current_tau[:, 2:3]
            
            energy = self.joint_energy(l1_current,l2_current,theta_current, log_jacob, c_target, r_target)
            
            # Total MALA Energy = Physical Energy - Log Determinant
            total_E = energy.sum()
            
            # 4. GRADIENT STEP
            grad = torch.autograd.grad(total_E, phi)[0]
            
            # 5. PROPOSAL (Langevin Step)
            noise = torch.randn_like(phi)
            phi_prop = phi - self.eta * grad + torch.sqrt(torch.tensor(2 * self.eta)) * noise
            
            # can perform Metropolis-Hastings Accept/Reject step here
            if mc_corrected:
                pass
            
            phi = phi_prop.detach().requires_grad_(True)

        # 6. FINAL OUTPUT: Convert the last phi back to physical tau
        with torch.no_grad():
            final_tau, _ = self.transform_to_physical_space(phi)
            
        return final_tau
                
"""