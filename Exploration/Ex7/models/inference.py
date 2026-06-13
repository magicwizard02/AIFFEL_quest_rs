import torch

def decode_sequence(input_seq, model, tar_word_to_index, tar_index_to_word, text_max_len, summary_max_len, device):
    """
    Predicts the summary for a given input sequence using the trained Seq2Seq model.
    Steps:
    1. Pass input through the Encoder.
    2. Start the Decoder with the <SOS> token.
    3. Iteratively generate the next word using Attention and the previous hidden state.
    """
    # 1. Prepare input: Ensure it's a tensor with batch dimension (1, seq_len)
    input_seq = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)

    model.eval() # Set model to evaluation mode (disables dropout)
    with torch.no_grad():
        # 2. Encoder pass
        encoder_outputs, h, c = model.encoder(input_seq)

    # 3. Initialize Target sequence with <SOS> token index
    target_seq = torch.zeros((1, 1), dtype=torch.long, device=device)
    target_seq[0, 0] = tar_word_to_index['sostoken']

    stop_condition = False
    decoded_sentence = ''

    # 4. Decoding Loop
    while not stop_condition:
        with torch.no_grad():
            # A. Single-step Decoder forward pass
            decoder_outputs, h, c = model.decoder(target_seq, h, c)

            # B. Apply Attention mechanism
            attn_out = model.attention(decoder_outputs, encoder_outputs)
            
            # C. Combine and Project to Vocabulary space
            combined = torch.cat((decoder_outputs, attn_out), dim=-1)
            combined = torch.tanh(model.concat(combined))
            output_tokens = model.output_layer(combined)

        # 5. Greedy Search: Pick the word with the highest probability
        sampled_token_index = torch.argmax(output_tokens[0, -1, :]).item()
        sampled_token = tar_index_to_word[sampled_token_index]

        if sampled_token != 'eostoken':
            decoded_sentence += ' ' + sampled_token

        # 6. Stop if <EOS> is reached or maximum length is exceeded
        if sampled_token == 'eostoken' or len(decoded_sentence.split()) >= (summary_max_len - 1):
            stop_condition = True

        # 7. Update target_seq for the next time step
        target_seq[0, 0] = sampled_token_index

    return decoded_sentence.strip()