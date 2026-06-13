import os
import torch
import pandas as pd

def save_weights(model, m_name, lr, batch, epoch, is_best=False):
    """
    Saves the model's state_dict (weights and biases).
    If is_best is True, it saves as 'best_model.pth' for the final evaluation.
    """
    weight_dir = os.path.join("results", m_name, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    
    filename = "best_model.pth" if is_best else f"weights_LR{lr}_B{batch}_epoch_{epoch}.pth"
    path = os.path.join(weight_dir, filename)
    
    torch.save(model.state_dict(), path)
    if is_best:
        print(f"[System] Best model weights updated at: {path}")

def load_weights(model, m_name, lr, batch, epoch, device='cpu', is_best=False):
    """
    Loads saved parameters into a model instance.
    """
    import os
    weight_dir = os.path.join("results", m_name, "weights")
    filename = "best_model.pth" if is_best else f"weights_LR{lr}_B{batch}_epoch_{epoch}.pth"
    path = os.path.join(weight_dir, filename)
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[System] Loaded weights from {path}")
        return model
    else:
        print(f"[Error] Weight file not found: {path}")
        return None

def update_results_refined(m_name, lr, batch, t_loss_list, t_acc, v_acc):
    """
    Logs batch-level loss and epoch-level accuracy into CSV files for tracking.
    """
    metric_dir = os.path.join("results", m_name, "metrics")
    os.makedirs(metric_dir, exist_ok=True)
    
    loss_file = os.path.join(metric_dir, f"loss_LR{lr}_B{batch}.csv")
    acc_file = os.path.join(metric_dir, f"accuracy_LR{lr}_B{batch}.csv")

    # Save detailed batch-by-batch loss
    new_loss_data = pd.DataFrame({"batch_loss": t_loss_list})
    new_loss_data.to_csv(loss_file, mode='a', header=not os.path.exists(loss_file), index=False)

    # Save high-level epoch accuracy
    new_acc_data = pd.DataFrame({"train_acc": [t_acc], "val_acc": [v_acc]})
    new_acc_data.to_csv(acc_file, mode='a', header=not os.path.exists(acc_file), index=False)

def get_model_at_stage(Encoder, Decoder, Seq2Seq, HP, shared_params, t_name, 
                       epoch_val, device, is_best_flag=False):
    """
    Helper to create a model shell and fill it using load_weights.
    Note: We pass the Classes (Encoder, Decoder, Seq2Seq) as arguments.
    """
    # Extracting values internally so you don't have to define them outside
    lr_val = HP["learning_rate"]
    batch_val = HP["batch_size"]

    # 1. Initialize Architecture
    enc = Encoder(vocab_size=HP["src_vocab_size"], **shared_params).to(device)
    dec = Decoder(vocab_size=HP["tar_vocab_size"], **shared_params).to(device)
    model_shell = Seq2Seq(enc, dec, HP["tar_vocab_size"], HP["hidden_size"]).to(device)
    
    # 2. Use the load_weights function defined above
    loaded_model = load_weights(
        model_shell, 
        t_name, 
        lr_val,    # Use extracted value
        batch_val, # Use extracted value
        epoch_val, 
        device=device, 
        is_best=is_best_flag
    )
    
    if loaded_model:
        loaded_model.eval()
        return loaded_model
    return None