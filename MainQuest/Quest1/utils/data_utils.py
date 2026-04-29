import torch

def pad_sequence(seq, max_len, pad_value=0):
    """
    Standardizes sequence lengths across a batch.
    Required for efficient parallel GPU computation by creating uniform tensor shapes.
    """
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))

def to_tensor(data_list, dtype=torch.long):
    """
    Transforms Python lists into PyTorch Tensors.
    Default dtype 'torch.long' is required for Embedding layers and CrossEntropyLoss.
    """    
    return torch.tensor(data_list, dtype=dtype)