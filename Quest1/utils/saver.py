import os
import torch
import pandas as pd

def save_weights(model, m_name, lr, batch, epoch, is_best=False):
    """
    Saves model weights to the structured directory.
    
    Structure: results/models/weights/{model_name}/
    - Saves 'best_model.pth' for peak performance.
    - Saves epoch-specific weights for checkpointing.
    """
    # Updated directory path: results -> models -> weights -> {m_name}
    weight_dir = os.path.join("results", "models", "weights", m_name)
    os.makedirs(weight_dir, exist_ok=True)
    
    filename = "best_model.pth" if is_best else f"weights_LR{lr}_B{batch}_epoch_{epoch}.pth"
    path = os.path.join(weight_dir, filename)
    
    # Save the state_dict (weights only)
    torch.save(model.state_dict(), path)
    if is_best:
        print(f"[System] Global best weights updated at: {path}")

def load_weights(model, m_name, device='cpu'):
    """
    Loads weights from the structured results directory.
    """
    # Matches the new structure: results/models/weights/{m_name}/
    path = os.path.join("results", "models", "weights", m_name, "best_model.pth")
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[System] Successfully restored parameters from {path}")
        return model
    else:
        print(f"[Error] Checkpoint not found at {path}")
        return None

def update_results_refined(m_name, lr, batch, t_loss, t_acc, v_acc):
    """
    Logs training and validation metrics into CSV files.
    
    Structure: results/models/metrics/{model_name}/
    """
    # Updated directory path: results -> models -> metrics -> {m_name}
    metric_dir = os.path.join("results", "models", "metrics", m_name)
    os.makedirs(metric_dir, exist_ok=True)
    
    acc_file = os.path.join(metric_dir, f"accuracy_LR{lr}_B{batch}.csv")

    # Handle case where t_loss might be a list (average it)
    avg_t_loss = sum(t_loss) / len(t_loss) if isinstance(t_loss, list) else t_loss

    # Append metrics to the CSV file
    acc_df = pd.DataFrame({
        "train_loss": [avg_t_loss], 
        "train_acc": [t_acc], 
        "val_acc": [v_acc]
    })
    
    # mode='a' (append) adds a new row each epoch
    acc_df.to_csv(acc_file, mode='a', header=not os.path.exists(acc_file), index=False)

