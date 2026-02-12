
"""

Loss = NLL of the data, where the likelihood is a Gaussian mixture

L = -ln(∑_K π_k(x) N(Γ|μ_k(x),σ_k(x)**2))

Your network would look like this:
    Input: Task vector [target,obstacle] 
    Hidden Layers: Standard ReLU/GELU layers.
    Output Layer: Three separate heads:
        Mixing Coefficients ($\pi_k$): Softmax activation (sums to 1).
        Means ($\mu_k$): Linear activation (the actual tool parameters).
        Variances ($\sigma_k$): Exponential or Softplus activation (must be positive).
    """

import torch
import numpy as np
