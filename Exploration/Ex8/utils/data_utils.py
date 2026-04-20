# ===========================================================================
# utils/data_utils.py
# Data Processing Utilities for Transformer Inputs
# ===========================================================================

import torch

# ===========================================================================
# 1. Teacher Forcing Data Slicer
# ===========================================================================

def create_decoder_data(ids):
    """
    Splits the answer sequence into Input and Target for Teacher Forcing.
    Shifted-right mechanism: Input drops last, Target drops first.

    Logic:
    - Input: Drop the last token (usually <EOS> or to match target length)
    - Target: Drop the first token (<BOS>)
    
    Example:
    Full: [BOS, I, am, happy, EOS]
    In:   [BOS, I, am, happy]
    Out:  [I, am, happy, EOS]
    """
    # Decoder Input (Shifted Right)
    decoder_input = ids[:-1]
    # Decoder Target (Ground Truth)
    decoder_target = ids[1:]
    
    return decoder_input, decoder_target


# ===========================================================================
# 2. Sequence Padding Utility
# ===========================================================================

def pad_sequence(seq, max_len, pad_value=0):
    """
    Ensures all sequences have a fixed length.
    Pads or truncates sequences to ensure fixed length for tensor batching.
    
    Args:
        seq: List of token IDs
        max_len: Desired sequence length
        pad_value: ID used for padding (default 0)
    """
    if len(seq) > max_len:
        # Truncate if sequence exceeds max_len
        return seq[:max_len]
    
    # Append pad_value until reaching max_len
    return seq + [pad_value] * (max_len - len(seq))


# ===========================================================================
# 3. Batch Preparation (Optional Helper)
# ===========================================================================

def to_tensor(data_list, dtype=torch.long):
    """
    Converts a list/series of padded sequences into a PyTorch LongTensor.
    """    
    return torch.tensor(data_list.tolist(), dtype=dtype)


import torch

# ===========================================================================
# 4. Tokenization & Inference Helpers
# ===========================================================================

def encode(text, sp_model):
    """
    Converts raw text into a list of subword token IDs using SentencePiece.
    
    Args:
        text (str): The input string.
        sp_model: The loaded SentencePieceProcessor instance.
    """
    return sp_model.encode_as_ids(text)


def decode(ids, sp_model):
    """
    Converts a list of token IDs back into a human-readable string.
    Automatically handles subword merging (e.g., 'I', ' am' -> 'I am').
    
    Args:
        ids (list): List of integer token IDs.
        sp_model: The loaded SentencePieceProcessor instance.
    """
    return sp_model.decode_ids(ids)


def add_special_tokens(token_ids, sp_model):
    """
    Wraps a token sequence with [BOS] at the start and [EOS] at the end.
    This is required for the Transformer to know where a sentence begins/ends.
    """
    return [sp_model.bos_id()] + token_ids + [sp_model.eos_id()]

def greedy_decode(model, sentence, sp_model, device, max_len=50):
    """
    Generates a response using the 'Greedy Search' algorithm.
    
    Inference Flow:
    1. The source sentence is encoded and sent to the device.
    2. The decoder starts with the [BOS] token.
    3. The model predicts the most likely next word (argmax).
    4. The word is appended to the input, and the process repeats until [EOS].
    """
    # Ensure model is in evaluation mode (no dropout)
    model.eval()

    # --- Step 1: Encoder Input Preparation ---
    # Convert input string to tokens and add special markers
    raw_ids = encode(sentence, sp_model)
    input_ids = add_special_tokens(raw_ids, sp_model)
    
    # Create tensor and add batch dimension [1, seq_len]
    src_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # --- Step 2: Decoder Initialization ---
    # We begin the generation with the Beginning-of-String ID
    generated_ids = [sp_model.bos_id()]
    eos_id = sp_model.eos_id()
    
    # --- Step 3: Auto-regressive Loop ---
    for _ in range(max_len):
        # Current generated sequence as a tensor
        tgt_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
        
        # We disable gradient calculation to speed up inference and save memory
        with torch.no_grad():
            # output shape: [batch, current_seq, vocab_size]
            output = model(src_tensor, tgt_tensor)
        
        # We take the logits for the very last word predicted
        # argmax(dim=-1) picks the index with the highest probability score
        next_token = output.argmax(dim=-1)[:, -1].item()
        
        generated_ids.append(next_token)
        
        # Stop generating if the model predicts the End-of-String ID
        if next_token == eos_id:
            break
            
    # --- Step 4: Final Output ---
    # Remove the special tokens and return the final string
    return decode(generated_ids, sp_model)