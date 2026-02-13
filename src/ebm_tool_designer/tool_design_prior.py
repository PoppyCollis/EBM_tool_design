"""
Unconditioned prior over tool designs. Sampling from this prior will give you random tools.
Right now it is independent factors of l1, l2, theta sampled from uniform distributions.
We will extend towards more expressive and complex priors in the future.

Note that we return sin(theta) and cos(theta) so that the network understands that 0 is near 360
(i.e. we deal with the wrap around problem).
"""

import numpy as np
from helpers.plots import visualise_tools

class ToolDesignPrior:
    def __init__(self, l1_range, l2_range, theta_range):
        """
        l1_range/l2_range: tuples (min, max)
        theta_range: tuple (min_deg, max_deg)
        """
        self.l1_bounds = l1_range
        self.l2_bounds = l2_range
        self.theta_bounds = (np.radians(theta_range[0]), np.radians(theta_range[1]))

    def sample_design(self, n_samples=1):
        """Samples tools using Uniform priors (simplest for EBM discovery)."""
        l1 = np.random.uniform(self.l1_bounds[0], self.l1_bounds[1], n_samples)
        l2 = np.random.uniform(self.l2_bounds[0], self.l2_bounds[1], n_samples)
        theta = np.random.uniform(self.theta_bounds[0], self.theta_bounds[1], n_samples)
        
        # Pre-calculate sin/cos for your Reward Network input
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        return {
            'l1': l1,
            'l2': l2,
            'theta': theta,
            'sin_theta': sin_theta,
            'cos_theta': cos_theta
        }

def main():
    # Define ranges [Lower, Upper]
    l1_range = (150,300)
    l2_range = (150,300)
    theta_range = (0,360)
    
    tool_prior = ToolDesignPrior(l1_range, l2_range, theta_range)

    num_designs = 12
    designs = tool_prior.sample_design(num_designs)

    visualise_tools(designs)
    
if __name__ == "__main__":
    main()