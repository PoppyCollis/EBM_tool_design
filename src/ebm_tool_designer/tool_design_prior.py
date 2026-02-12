"""
Unconditioned prior over tool designs. Sampling from this prior will give you random tools.
"""

import numpy as np
from helpers.plots import visualise_tools

class ToolDesignPrior:
    def __init__(self, l1_mu, l1_sigma, l2_mu, l2_sigma, theta_mu, theta_sigma):
        """
        Initialises the Gaussian priors for the tool parameters.
        """
        self.priors = {
            'l1': (l1_mu, l1_sigma),
            'l2': (l2_mu, l2_sigma),
            'theta': (np.radians(theta_mu), np.radians(theta_sigma))
        }

    def sample_design(self, n_samples=1):
        """Samples n tool designs from the Gaussian priors."""
        samples = {}
        for param, (mu, sigma) in self.priors.items():
            samples[param] = np.random.normal(mu, sigma, n_samples)
        
        # Ensure lengths are positive (clipping at a small epsilon)
        samples['l1'] = np.maximum(samples['l1'], 0.1)
        samples['l2'] = np.maximum(samples['l2'], 0.1)
        
        return samples

def main():
    # Define priors: (mean, std_dev)
    model = ToolDesignPrior(200, 50, 200, 50, 180, 90)

    # Generate 5 random designs
    num_designs = 12
    designs = model.sample_design(num_designs)
    visualise_tools(designs)
    
if __name__ == "__main__":
    main()