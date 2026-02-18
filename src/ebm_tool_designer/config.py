import numpy as np
import torch

class ToolDatasetConfig:
    
    L1_BOUNDS = (150.0,300.0) # make sure these are floats
    L2_BOUNDS = (150.0,300.0)
    THETA_BOUNDS = (0.0,360.0)
    
    NUM_DESIGNS = 1000
    
    REWARD_TYPE = "euclidean_distance"
    
    SAVE_PATH = 'src/ebm_tool_designer/data/dummy_dataset.parquet'
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    
    
    
class RewardModelConfig:
    
    NUM_SEEDS = 5
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    DATA_PATH = 'src/ebm_tool_designer/data/dummy_dataset.parquet'
    WEIGHTS_SAVE_PATH = 'src/ebm_tool_designer/weights/reward_model_best.pt'
    
    IN_FEATURES = 6
    HIDDEN_FEATURES = 128
    OUT_FEATURES = 64
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 10
    
    SIGMA = 0.01 # 0.01 after normalizing all the continuous attributes to [0, 1]
    
    
class EBMConfig:
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SIGMA = 1e-2
    SAMPLING_METHOD = 'langevin'
    N_SAMPLES = 10
    N_SAMPLING_STEPS = 50
    ETA = 1e-4

