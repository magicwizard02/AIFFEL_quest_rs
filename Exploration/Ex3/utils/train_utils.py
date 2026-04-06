import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# [1] LOGGER: Consolidates metrics into CSV files
# ---------------------------------------------------------------------------
def update_results_refined(m_name, lr, batch, t_loss_list, t_acc, v_acc):
    """
    Saves training metrics into a hierarchical directory.
    - Loss: Detailed batch-by-batch data.
    - Accuracy: Summary per epoch.
    """
    metric_dir = os.path.join("results", m_name, "metrics")
    os.makedirs(metric_dir, exist_ok=True)
    
    loss_file = os.path.join(metric_dir, f"loss_LR{lr}_B{batch}_detailed.csv")
    acc_file = os.path.join(metric_dir, f"accuracy_LR{lr}_B{batch}_summary.csv")

    # Part A: Append Batch-Level Loss
    new_loss_data = pd.DataFrame({"batch_loss": t_loss_list})
    new_loss_data.to_csv(loss_file, mode='a', header=not os.path.exists(loss_file), index=False)

    # Part B: Append Epoch-Level Accuracy
    new_acc_data = pd.DataFrame({
        "train_acc": [t_acc] if not isinstance(t_acc, list) else t_acc,
        "val_acc": [v_acc] if not isinstance(v_acc, list) else v_acc
    })
    new_acc_data.to_csv(acc_file, mode='a', header=not os.path.exists(acc_file), index=False)

# ---------------------------------------------------------------------------
# [2] WEIGHT SAVER: Saves model state_dict
# ---------------------------------------------------------------------------
def save_weights(model, m_name, lr, batch, epoch):
    """
    Saves the model weights to the 'weights' subdirectory.
    """
    weight_dir = os.path.join("results", m_name, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    
    path = os.path.join(weight_dir, f"weights_LR{lr}_B{batch}_epoch_{epoch}.pth")
    torch.save(model.state_dict(), path)
    print(f"[System] Saved weights to {path}")

# ---------------------------------------------------------------------------
# [3] WEIGHT LOADER: For resuming training (Time Machine)
# ---------------------------------------------------------------------------
def load_weights(model, m_name, lr, batch, epoch):
    """
    Loads weights from a specific epoch to resume training.
    """
    path = f"results/{m_name}/weights/weights_LR{lr}_B{batch}_epoch_{epoch}.pth"
    
    if os.path.exists(path):
        # map_location ensures it works on CPU or MPS/GPU
        model.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"[System] Resumed from {path}")
        return model
    else:
        print(f"[Error] Weight file not found: {path}")
        return None

# ---------------------------------------------------------------------------
# [4] METRIC RETRIEVER: For plotting later
# ---------------------------------------------------------------------------
def load_refined_metric(m_name, metric_type, lr=0.001, batch=32):
    """
    Retrieves logged data from CSV for visualization.
    """
    metric_dir = os.path.join("results", m_name, "metrics")
    
    if metric_type == 'loss':
        fname = f"loss_LR{lr}_B{batch}_detailed.csv"
        target_col = 'batch_loss'
    else:
        fname = f"accuracy_LR{lr}_B{batch}_summary.csv"
        target_col = 'val_acc'
        
    fpath = os.path.join(metric_dir, fname)
    return pd.read_csv(fpath)[target_col].tolist() if os.path.exists(fpath) else []

# -------------------------------------------------------------------------
# [5] Transform Wrapper
# -------------------------------------------------------------------------
class ApplyTransform(Dataset):
    '''
    A Wrapper to apply a specific transform to a Dataset subset.
    '''
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# -------------------------------------------------------------------------
# [5-a]: Define the Base Transform (Mandatory Preprocessing)
# -------------------------------------------------------------------------
def get_base_transform():
    '''
    Standard preprocessing for all images to match ResNet50 input requirements.
    Stats Explained:
    - Calculated Mean and Standard Deviation of the millions of images in ImageNet.
    - 0.485, 0.456, 0.406: Average Red, Green, and Blue intensities.
    '''
    return transforms.Compose([
        # Resize the image to 224x224 to unify the input dimensions
        transforms.Resize((224, 224)), 
        # Convert the PIL image to a PyTorch Tensor [0.0, 1.0]
        transforms.ToTensor(),         
        # Normalize using ImageNet statistics for better convergence
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
