import numpy as np
import torch

class ToolDatasetConfig:
    
    L1_BOUNDS = (150,300)
    L2_BOUNDS = (150,300)
    THETA_BOUNDS = (0,360)
    
    NUM_DESIGNS = 1000
    
    REWARD_TYPE = "euclidean_distance"
    
    SAVE_PATH = 'src/ebm_tool_designer/data/dummy_dataset.parquet'
    
    
    
class RewardModelConfig:
    
    NUM_SEEDS = 5
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    DATA_PATH = 'src/ebm_tool_designer/data/dummy_dataset.parquet'
    WEIGHTS_SAVE_PATH = 'src/ebm_tool_designer/weights/reward_model_best.pt'
    
    HIDDEN_FEATURES = 128
    OUT_FEATURES = 64
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 10
    
    
class EBMConfig:
    SIGMA = 0.01
    SAMPLING_METHOD = 'langevin'
    N_SAMPLES = 100

