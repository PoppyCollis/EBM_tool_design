"""
Convert tool design prior and reward prediction model into an energy-based model which we can sample from.
"""
import numpy as np
from reward_model import MLP
import torch
from tool_design_prior import ToolDesignPrior
from config import ToolDatasetConfig

    
class EnergyBasedModel:    
    def __init__(self, prior, weights_path):
        """
        Initializes the EBM with a prior and a reward model.
        
        Args:
            prior (ToolDesignPrior): The unconditioned prior over tool designs.
            weights_path (str): Path to the pre-trained reward model weights.
        """
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = torch.nn.MSELoss()
        self.prior = prior
        # load in pre-trained reward model weights
        self.reward_model = self.load_reward_model(weights_path)
        self.sigma = 0.01
        
    def load_reward_model(self, weights_path):
        # load in architecture (must match one used for pre-trained model weights)
        reward_model = MLP() 
        
        # load the pre-trained weights into the model
        checkpoint = torch.load(weights_path)
        reward_model.load_state_dict(checkpoint['model_state_dict'])
        reward_model.eval()
        reward_model.to(self.device)
        
        return reward_model
    
    def reward_prediction_error(self, tool_description, target_location, reward_target):
        with torch.no_grad(): 
            rpes = []
            for features, labels in tool_description:
                features = features.to(self.device)
                reward_target = reward_target.to(self.device)
                preds = self.reward_model(features)
                loss = self.criterion(preds, reward_target)
                rpes.append(loss.item())
        return rpes
            

    def energy(self, tool_description, target_location, reward_target):
        """
        Calculates the energy for a given set of designs and task.
        """        
        E = np.inv(2 * self.sigma**2)* self.reward_prediction_error(self, tool_description, reward_target)
        E = E # + np.log(prior)
        return E
    

    def sample(self, target_location, n_samples=1, method='langevin'):
        """
        Samples from the EBM distribution p(τ | c) ∝ exp(-E(τ, c)).
        """
        if method == 'langevin':
            pass
        elif method == "MCMC":
            pass
        else:
            raise NotImplementedError("Sampling method not supported.")
        

def main():
    
    weights_path = ToolDatasetConfig.WEIGHTS_SAVE_PATH
    
    # Define ranges [Lower, Upper]
    l1_bounds = ToolDatasetConfig.L1_BOUNDS
    l2_bounds = ToolDatasetConfig.L2_BOUNDS
    theta_bounds = ToolDatasetConfig.THETA_BOUNDS

    prior = ToolDesignPrior(l1_bounds, l2_bounds, theta_bounds)
    
    ebm = EnergyBasedModel(prior, weights_path=weights_path)

if __name__ == "__main__":
    main()