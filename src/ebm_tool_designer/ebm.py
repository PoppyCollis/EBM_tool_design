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
        # also need to load in normalisation stats for features
        reward_model.eval()
        reward_model.to(self.device)
        return reward_model
    
    def joint_energy(self, l1, l2, theta, c_target, r_target):
        # current sample, \tau, c, reward target
        # conditional energy + prior energy [batch_size, 6]
        si = torch.sin(theta)
        co = torch.cos(theta)
        
        x = torch.cat([l1, l2, si, co, c_target], dim=-1)
        
        # 3. Apply the same normalization used during training
        # x = (x - self.feature_stats['mean']) / self.feature_stats['std']
        
        cond_energy = self.reward_model.energy(x, r_target)
        
        prior_energy = 0.0
        
        return cond_energy + prior_energy
    
    def transform_to_unconstrained_space(self, tau):
        """ Physical [low, high] -> Unconstrained space (-inf, inf)"""
        bounds_low = torch.tensor([self.prior.l1_bounds[0], self.prior.l2_bounds[0], self.prior.theta_bounds[0]]).to(self.device)
        bounds_high = torch.tensor([self.prior.l1_bounds[1], self.prior.l2_bounds[1], self.prior.theta_bounds[1]]).to(self.device)
        # Normalize to [0, 1]
        p_norm = (tau - bounds_low) / (bounds_high - bounds_low)
        # Apply Logit: log(p / (1-p))
        # We add a tiny epsilon to prevent log(0)
        eps = 1e-6
        p_norm = torch.clamp(p_norm, eps, 1.0 - eps)
        phi = torch.log(p_norm / (1.0 - p_norm))
        return phi
    
    def transform_to_physical_space(self, phi):
        """Unconstrained (-inf, inf) -> Physical [low, high]"""
        bounds_low = torch.tensor([self.prior.l1_bounds[0], self.prior.l2_bounds[0], self.prior.theta_bounds[0]]).to(self.device)
        bounds_high = torch.tensor([self.prior.l1_bounds[1], self.prior.l2_bounds[1], self.prior.theta_bounds[1]]).to(self.device)
        s = torch.sigmoid(phi)
        tau = self.bounds_low + (bounds_high - bounds_low) * s
        
        # Calculate Log-Jacobian: log|d_tau / d_phi|
        # This is required for MALA to stay mathematically 'correct'
        log_det = torch.log(bounds_high - bounds_low) + \
                torch.log(s + 1e-6) + torch.log(1.0 - s + 1e-6)
        
        return tau, log_det.sum(dim=-1)

    def langevin_dynamics(self, c_target, r_target, batch_size=32, mc_corrected=False):
        
        # Logic:
        
        # 1. sample from tool design prior (uniform distributions)
        # 2. immediately reparametrise into phi space
        # 3. calculate the gradient of the energy with respect to phi.
        # LD "proposal" step: to move phi_t -> phi_t+1.
        #  Convert \phi_t+1 back to tau_t+1 to calculate the Metropolis-Hastings acceptance ratio.
        
        # 1. INITIALISE tool params, tau, by sampling from the prior (e.g., [batch_size, 4])
        
        l1_init,l2_init,theta_init = self.prior.sample(batch_size)
        
        tau = torch.stack([l1_init, l2_init, theta_init], dim=-1).to(self.device).requires_grad_(True)
        
        c_target = c_target.to(self.device).detach()
        
        # 2. TRANSFORM tau by converting into unconstrained space (phi) using sigmoid
        
        phi = self.transform_to_unconstrained_space(tau)

        for i in range(self.n_sampling_steps):
            
            # 3. ENERGY CALCULATION
        
            # Convert phi back to physical tau inside the gradient tape
            current_tau, log_jacob = self.transform_to_physical_space(phi)
            
            # Calculate energy using your existing joint_energy logic
            # slice tau to get individual components for the Energy logic (which works on theta not sin(theta))
            l1_current = current_tau[:, 0:1]
            l2_current = current_tau[:, 1:2]
            theta_current = current_tau[:, 2:3]
            
            energy = self.joint_energy(l1_current,l2_current,theta_current, c_target, r_target)
            
            # Total MALA Energy = Physical Energy - Log Determinant
            total_E = energy.sum() - log_jacob.sum()
            
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
                

prior = ToolDesignPrior(ToolDatasetConfig.L1_BOUNDS, ToolDatasetConfig.L2_BOUNDS, ToolDatasetConfig.THETA_BOUNDS)
        
ebm = EnergyBasedModel(prior, weights_path=RewardModelConfig.WEIGHTS_SAVE_PATH)
    


# tests: 
# 1. get two tool vectors and work out the joint energy of each
# 2. check energy is higher when reward pred is further from target
# 3. 


# CONSIDERATIONS:

# The $\sin(\theta)^2 + \cos(\theta)^2 = 1$ Constraint. sample 3: $(l_1, l_2, \theta)$, This guarantees the angle is always valid.

# the reward model was trained on normalised data: apply feature_stats to tau and c inside the joint energy function - 
# If your reward model was trained on normalized data, you should be performing Langevin in the normalized space.

# MALA combines Langevin Dynamics with a Metropolis-Hastings acceptance step. 
