import matplotlib.pyplot as plt
import numpy as np
    
def visualise_tools(designs):
    n_samples = len(designs["l1"])
    
    # We need to store the actual (x, y) points to determine the plot bounds
    all_joint_positions = [] 
    
    for i in range(n_samples):
        l1 = designs['l1'][i]
        l2 = designs['l2'][i]
        theta = designs['theta'][i]
        
        # Calculate vertical orientation coordinates
        p1 = (0, 0)           # Base
        p2 = (0, l1)          # Joint (Vertical)
        p3 = (l2 * np.sin(theta), l1 + l2 * np.cos(theta)) # Tip
        
        all_joint_positions.append([p1, p2, p3])

    # Convert to a flat numpy array of (x, y) pairs to find the global max/min
    # Shape will be (n_samples * 3, 2)
    all_pts = np.array(all_joint_positions).reshape(-1, 2)
    
    # Calculate bounds based on the maximum excursion in any direction
    max_val = np.max(np.abs(all_pts)) * 1.1 
    
    # Set up subplots
    cols = 3
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False, constrained_layout=True)
    axes = axes.flatten()

    for i in range(n_samples):
        p1, p2, p3 = all_joint_positions[i]
        ax = axes[i]
        
        # Extract x and y lists for plotting
        xs = [p1[0], p2[0], p3[0]]
        ys = [p1[1], p2[1], p3[1]]
        
        ax.plot(xs, ys, '-o', lw=4, markersize=2, color='#2c3e50')
        
        # Consistent scaling across all subplots
        ax.set_title(f"θ: {np.degrees(designs['theta'][i]):.1f}°\nl1: {designs['l1'][i]:.1f}\nl2: {designs['l2'][i]:.1f}")
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val) 
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()
    
    
def plot_losses(epochs, epoch_train_losses, epoch_val_losses, title= "Loss vs Epochs"):  
    
    plt.plot(range(epochs), epoch_train_losses, label='Train')
    plt.plot(range(epochs), epoch_val_losses, label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(title)
    
    plt.show()
    
    
def plot_mean_losses(epochs, mean_train_loss, std_train_loss, mean_val_loss, std_val_loss, title= "Mean Loss vs Epochs"):
    x_axis = range(epochs)
    
    plt.plot(x_axis, mean_train_loss, label='Mean train Loss')
    plt.fill_between(x_axis, 
                 mean_train_loss - std_train_loss, 
                 mean_train_loss + std_train_loss, 
                 alpha=0.2, label='$\pm$ 1 Std Dev')
    
    plt.plot(x_axis, mean_val_loss, label='Mean val Loss')
    plt.fill_between(x_axis, 
                 mean_val_loss - std_val_loss, 
                 mean_val_loss + std_val_loss, 
                 alpha=0.2, label='$\pm$ 1 Std Dev')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title(title)
    plt.show()