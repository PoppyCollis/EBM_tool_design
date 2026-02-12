"""
Convert tool design prior and reward prediction model into an energy-based model which we can sample from.
"""
import numpy as np


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

    def energy(self, designs, task_description):
        """
        Calculates the energy for a given set of designs and task.
        E(τ) = 1/2 σ^2 (r-fθ(τ, c))^2 + 1/2 ∥τ ∥_2^2
        
        Args:
            designs (dict): Dictionary of design parameters (l1, l2, theta).
            task_description: The context/task vector c.
        """        
        # Get reward prediction
        reward = self.reward_model.predict(designs, task_description)
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
