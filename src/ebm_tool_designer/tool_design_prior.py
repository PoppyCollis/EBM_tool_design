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

        return l1,l2, theta
    
    
    def transform_to_unconstrained(params, bounds):
        # transform bounds into unconstrained space via reparameterisation into phi
        # [0,1] to Sigmoid
        pass
    
    def transform_to_physical(phi, bounds):
        # transform Phi back to original parameter space
        # Sigmoid back to [0, 1]
        pass