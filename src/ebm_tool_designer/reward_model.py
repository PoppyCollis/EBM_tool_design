"""
Neural network reward prediction model f(τ, c) → R; takes in tool design parameters τ 
and task description c and outputs a scalar reward
"""

import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from tool_dataset import CustomDataset
from helpers.plots import plot_losses, plot_mean_losses


class MLP(nn.Module):
    def __init__(self, in_features=6, hidden_features=128, out_features=64):
        
        # Use super __init__ to inherit from parent nn.Module class
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, 1)
        )
        # Apply the weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Kaiming Uniform is ideal for ReLU activations
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        return self.net(x)
    
    
def train_model(device, reward_model, train_loader, test_loader, optimizer, criterion, epochs):
    reward_model.train()
    reward_model.to(device) # Move model once
    
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(device)
            labels = labels.to(device)
            # Standard training loop logic here...
            optimizer.zero_grad()
            r_pred = reward_model.forward(features)
            loss = criterion(r_pred, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_train_losses.append(epoch_loss)
        
        reward_model.eval()
        val_loss = 0.0
        
        with torch.no_grad(): # Disable gradient calculation (saves memory/time)
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                preds = reward_model(features)
                loss = criterion(preds, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        epoch_val_losses.append(avg_val_loss)
        
        reward_model.train()
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    #plot_losses(epochs, epoch_train_losses, epoch_val_losses)
        
    return epoch_train_losses, epoch_val_losses

def run_n_trials(num_trials, device, train_loader, test_loader, criterion, epochs, lr):
    
    train_losses_list = []
    val_losses_list = []
    for i in range(num_trials):
        print(f"Starting Seed {i+1}/5")
        reward_model = MLP(in_features=6).to(device)
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
        train_losses, val_losses = train_model(device, reward_model, train_loader, test_loader, optimizer, criterion, epochs=epochs)
        train_losses_list.append(train_losses)
        val_losses_list.append(val_losses)
    
    mean_train_loss = np.mean(np.array(train_losses_list), axis=0) 
    mean_val_loss = np.mean(np.array(val_losses_list), axis=0)
    std_train_loss = np.std(np.array(train_losses_list), axis=0)
    std_val_loss = np.std(np.array(val_losses_list), axis=0)
    
    return mean_train_loss, mean_val_loss, std_train_loss, std_val_loss
    
    

def main():
    
    batch_size = 32
    epochs = 10
    lr = 1e-3
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    
    # Read data
    data_path = 'src/ebm_tool_designer/data/dummy_dataset.parquet'
    full_df = pd.read_parquet(data_path)
    
    # train test split 80%
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
    
    cols = ['l1', 'l2', 'sin_theta', 'cos_theta', "x_target", "y_target"]
    label_col = ['reward']
    
    # Calculate mean and std of features: should only calculate them based on the train portion of data
    mean_vals = torch.tensor(train_df[cols].mean().values, dtype=torch.float32)
    std_vals = torch.tensor(train_df[cols].std().values, dtype=torch.float32)
    
    # Don't normalize sin(theta) and cos(theta)
    # We set mean to 0 and std to 1 for those specific indices
    mean_vals[2:4] = 0.0
    std_vals[2:4] = 1.0
    
    feature_stats = {'mean': mean_vals, 'std': std_vals}

    # Calculate mean and std of rewards
    label_stats = {
    'mean': torch.tensor(train_df[label_col].mean().values, dtype=torch.float32),
    'std': torch.tensor(train_df[label_col].std().values, dtype=torch.float32)
    }
    
    train_dataset = CustomDataset(train_df, feature_stats, label_stats)
    test_dataset = CustomDataset(test_df, feature_stats, label_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    reward_model = MLP()
    # Use Mean Squared Error loss for regression
    criterion = nn.MSELoss()
    # Adam is a good general purpose optimizer
    optimizer = torch.optim.Adam(reward_model.parameters(), lr = lr)
    
    # Run 5 seeds of training sessions and look at the average train and val losses
    num_trials = 5

    mean_train_loss, mean_val_loss, std_train_loss, std_val_loss = run_n_trials(num_trials, device, train_loader, test_loader, criterion, epochs, lr)
    
    plot_mean_losses(epochs, mean_train_loss, std_train_loss, mean_val_loss, std_val_loss)
    
    
    # --- SAVING ---
    # Best practice: Save the model state, but also the optimizer state and metadata
    save_path = "src/ebm_tool_designer/weights/reward_model_best.pt"

    torch.save({
         'epoch': epochs,
         'model_state_dict': reward_model.state_dict(),
         'feature_stats': feature_stats, # VERY IMPORTANT: Save your normalization stats!
         'label_stats': label_stats,
     }, save_path)


if __name__ == "__main__":
    main()    