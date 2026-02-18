import numpy as np
import torch
from config import ToolDatasetConfig

class ToolDesignPrior:
    def __init__(self, l1_bounds, l2_bounds, theta_bounds, device):
        self.device = device
        self.bounds_low = torch.tensor([l1_bounds[0], l2_bounds[0], theta_bounds[0]], requires_grad=True).to(device)
        self.bounds_high = torch.tensor([l1_bounds[1], l2_bounds[1], theta_bounds[1]], requires_grad=True).to(device)
        
        
    def sample(self, batch_size=1):
        # sample in a way that will allow pytorch to track gradients
        n_params = 3
        
        epsilon = torch.rand(batch_size, n_params, device=self.device) # batch sample
        tau = self.bounds_low + (self.bounds_high - self.bounds_low) * epsilon
        
        tau.requires_grad_(True)
        return tau
    
    def transform_to_phi(self, tau):
        """
        Reparametrisation trick for tau into an unconstrained space phi by normalising and applying the logit func.
        This maps into a space between -infty, +infty such that we can do gradient update without the constraint 
        of hard bounds [a,b] of a uniform distribution, which causes problems with infinite gradients at the bounds.
        """
       # tau must be a tensor
        # Step 1: Normalize to [0, 1]
        u = (tau - self.bounds_low) / (self.bounds_high - self.bounds_low)
        # add a tiny epsilon to prevent log(0)
        eps = 1e-8
        u = torch.clamp(u, eps, 1.0 - eps)
        # transform to between (-\infty, + \infty)
        phi = torch.logit(u).detach().requires_grad_(True)
        return phi
        
    def transform_to_tau(self, phi):
        # phi must be a tensor, with  requires_grad=True
        tau = self.bounds_low + (self.bounds_high - self.bounds_low) * torch.sigmoid(phi)
        # do I need to return the log determinant here for MALA later?
        return tau

    
    
def main():
    
    """
    Here, I test that the reparameterisation trick works by intrucing a dummy energy function E = tau **2.
    When I optimise in phi space, the final tau parameteres are the lower bounds "a" of all of the uniform distrubitions.
    This is exactly what we would expect.
    """
    prior = ToolDesignPrior(l1_bounds=ToolDatasetConfig.L1_BOUNDS, l2_bounds=ToolDatasetConfig.L2_BOUNDS, theta_bounds=ToolDatasetConfig.THETA_BOUNDS, device=ToolDatasetConfig.DEVICE)

    # 1. start with a sample in tau space
    tau = prior.sample(batch_size=2)
    print(tau)

    # 2. translate to phi space
    phi = prior.transform_to_phi(tau) # now we are tracking grads, phi is the leaf node
    print(phi)

    # 3. optimise in phi space
    optimizer = torch.optim.SGD([phi], lr=1e-2) # will likely need Adam for a more complex energy function than this dummy one

    for i in range(10):
        optimizer.zero_grad() 
        
        tau_current = prior.transform_to_tau(phi) # re-derive tau from phi inside the loop.

        energy = torch.sum(tau_current ** 2) # dummy energy function
        
        # backprop
        energy.backward()
        
        optimizer.step()


    print(tau_current)
    
if __name__ == "__main__":
    main()  

