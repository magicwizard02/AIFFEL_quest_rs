
from torch.nn.utils.rnn import pad_sequence 
import torch

def sp_tokenize(sp_processor, corpus, max_len=None):
    """
    Function to tokenize a corpus using a trained SentencePiece processor.
    Matches the structural output of the provided baseline 'tokenize' function.
    """
    tensor = []

    # 1. Encode sentences into ID sequences
    for sen in corpus:
        # We ensure sen is a string and encode it to IDs
        tensor.append(torch.tensor(sp_processor.EncodeAsIds(str(sen))))

    # 2. Extract Vocabulary (word_index and index_word)
    word_index = {}
    index_word = {}
    
    # Access the vocabulary from the trained model's vocab file
    # Ensure this matches the model_prefix used in STEP 5
    vocab_file = 'nsmc_unigram_8k.vocab'
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            word = line.split("\t")[0]
            word_index[word] = idx
            index_word[idx] = word

    # 3. Padding logic using PyTorch's pad_sequence
    # This aligns sequences to the length of the longest sentence in the batch
    tensor = pad_sequence(tensor, batch_first=True, padding_value=0)

    # 4. Force fixed max_len for model training consistency
    if max_len:
        if tensor.size(1) > max_len:
            tensor = tensor[:, :max_len]
        else:
            pad_size = max_len - tensor.size(1)
            padding = torch.zeros((tensor.size(0), pad_size), dtype=torch.long)
            tensor = torch.cat([tensor, padding], dim=1)

    return tensor, word_index, index_word