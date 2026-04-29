import os
import torch
import pandas as pd

def save_weights(model, m_name, lr, batch, epoch, is_best=False):
    """
    Saves the model's parameters to the results directory.
    
    Args:
        model: The GPT model instance.
        m_name (str): Model/Experiment name (e.g., 'GPT_Pretraining' or 'GPT_SFT').
        lr (float): Learning rate used.
        batch (int): Batch size used.
        epoch (int): Current epoch number.
        is_best (bool): If True, saves as 'best_model.pth' for easy access.
    """
    weight_dir = os.path.join("results", m_name, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    
    filename = "best_model.pth" if is_best else f"weights_LR{lr}_B{batch}_epoch_{epoch}.pth"
    path = os.path.join(weight_dir, filename)
    
    torch.save(model.state_dict(), path)
    if is_best:
        print(f"[System] Best model weights updated at: {path}")


def load_weights(model, path, device='cpu'):
    """
    Loads weights into the model. 
    Crucial for:
    1. Resuming interrupted training.
    2. Loading Pretrained weights before starting SFT.
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[System] Successfully loaded weights from {path}")
        return model
    else:
        print(f"[Error] Weight file not found: {path}")
        return None


def update_results(m_name, lr, batch, t_loss_list, t_acc, v_acc):
    """
    Logs training metrics into CSV files for performance visualization.
    """
    metric_dir = os.path.join("results", m_name, "metrics")
    os.makedirs(metric_dir, exist_ok=True)
    
    loss_file = os.path.join(metric_dir, f"loss_LR{lr}_B{batch}.csv")
    acc_file = os.path.join(metric_dir, f"accuracy_LR{lr}_B{batch}.csv")

    # Save batch-level loss (to see the learning curve within epochs)
    new_loss_data = pd.DataFrame({"batch_loss": t_loss_list})
    new_loss_data.to_csv(loss_file, mode='a', header=not os.path.exists(loss_file), index=False)

    # Save epoch-level accuracy (to track generalization)
    new_acc_data = pd.DataFrame({"train_acc": [t_acc], "val_acc": [v_acc]})
    new_acc_data.to_csv(acc_file, mode='a', header=not os.path.exists(acc_file), index=False)


def get_lr_lambda(d_model, warmup_steps=4000):
    """
    Standard Transformer LR scheduler lambda function.
    Scale the learning rate based on the model dimension and warmup steps.
    """
    def lr_lambda(step):
        # Step starts from 0, so we add 1 to avoid division by zero
        step = max(1, step)
        
        # Original formula: d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
        arg1 = step ** -0.5
        arg2 = step * (warmup_steps ** -1.5)
        
        return (d_model ** -0.5) * min(arg1, arg2)
    
    return lr_lambda


