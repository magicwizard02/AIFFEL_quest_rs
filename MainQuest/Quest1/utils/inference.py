import torch

def greedy_decode_gpt(model, sp, prompt, device, max_len=50):
    """
    Generates a response from the GPT model token-by-token.
    
    Args:
        model: The trained GPT model.
        sp: The SentencePiece processor.
        prompt (str): The user's input (e.g., "<s> [User] Hello [Assistant] ").
        device: CPU/GPU/MPS.
        max_len (int): Maximum number of new tokens to generate.
    """
    model.eval()
    
    # 1. Encode the prompt into IDs
    # We use sp.encode_as_ids directly as you preferred
    input_ids = sp.encode_as_ids(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 2. Autoregressive Generation Loop
    for _ in range(max_len):
        # Create mask for the current sequence length
        # We import the mask function here or pass it as an argument
        from utils.masking import create_look_ahead_mask
        mask = create_look_ahead_mask(input_tensor, pad_id=sp.pad_id())
        
        with torch.no_grad():
            # Forward pass: logits shape [1, current_seq_len, vocab_size]
            logits = model(input_tensor, mask)
            
            # 3. Get the last token's predictions
            # We only care about the very last word predicted
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # 4. Append the predicted token to the input
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            # 5. Stop if the model predicts the End-of-String (EOS) token
            if next_token.item() == sp.eos_id():
                break
    
    # 6. Decode the final ID sequence back to text
    generated_ids = input_tensor.squeeze().tolist()
    return sp.decode_ids(generated_ids)